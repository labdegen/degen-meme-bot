import os
import re
import time
import glob
import logging
import asyncio
import requests
import signal
import random

from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv

import tweepy
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

# ——— CONFIG & SETUP —————————————————————————————

# Load environment variables
load_dotenv()

# Configure logging for Render's environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate required environment variables
REQUIRED_ENV_VARS = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_URL"  # Changed to REDIS_URL for Render compatibility
]

missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

# Twitter API setup (v1.1 and v2)
oauth = tweepy.OAuth1UserHandler(
    os.getenv("X_API_KEY"),
    os.getenv("X_API_KEY_SECRET"),
    os.getenv("X_ACCESS_TOKEN"),
    os.getenv("X_ACCESS_TOKEN_SECRET")
)
x_api = tweepy.API(oauth)

x_client = tweepy.Client(
    bearer_token=os.getenv("X_BEARER_TOKEN"),
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_KEY_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. "
    f"Always speak about the $DEGEN token at contract address {DEGEN_ADDR}. "
    "Do NOT mention any other token or chain."
)

# Regular expressions for address and symbol matching
ADDR_RE = re.compile(r"\b[A-Za-z0-9]{43,44}\b")
SYMBOL_RE = re.compile(r"\$([A-Za-z0-9]{2,10})", re.IGNORECASE)

# Rate limits & queues
RATE_WINDOW = 900  # 15 minutes in seconds
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
SEARCH_LIMIT = 10

mentions_q = deque(maxlen=MENTIONS_LIMIT+1)
tweets_q = deque(maxlen=TWEETS_LIMIT+1)
search_q = deque(maxlen=SEARCH_LIMIT+1)

# since_id workaround (48 hours ago)
INITIAL_SEARCH_ID = str(((int(time.time()*1000) - 1_728_000_000) << 22))

# Render specific: Add graceful shutdown handling
shutdown_event = asyncio.Event()

# ——— REDIS CONNECTION —————————————————————————————

@asynccontextmanager
async def get_redis_connection():
    """Context manager for Redis connections with automatic reconnection."""
    redis_url = os.getenv("REDIS_URL")
    client = None
    try:
        # Parse Redis URL for Render compatibility
        client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        await asyncio.to_thread(client.ping)
        logger.info("🔑 Redis connected")
        yield client
    except RedisConnectionError:
        logger.error("❌ Redis connection failed - will retry")
        if client:
            await asyncio.to_thread(client.close)
        await asyncio.sleep(5)
        raise
    except Exception as e:
        logger.error(f"❌ Redis error: {e}")
        if client:
            await asyncio.to_thread(client.close)
        raise
    finally:
        if client:
            await asyncio.to_thread(client.close)

# Global redis client - will be initialized in main()
redis_client = None

# ——— UTILITIES ———————————————————————————————————

def truncate_to_sentence(text: str, max_length: int) -> str:
    """Truncate text to the nearest sentence end that fits within max_length."""
    if len(text) <= max_length:
        return text
    s = text[:max_length]
    for sep in (". ", "! ", "? "):
        idx = s.rfind(sep)
        if idx != -1:
            return s[:idx+1]
    return s

async def get_thread_key(cid): 
    return f"{REDIS_PREFIX}thread:{cid}"

async def get_thread_history(cid): 
    key = await get_thread_key(cid)
    return await asyncio.to_thread(redis_client.hget, key, "history") or ""

async def increment_thread(cid):
    key = await get_thread_key(cid)
    pipe = redis_client.pipeline()
    pipe.hincrby(key, "count", 1)
    pipe.expire(key, 86400)  # 24 hour TTL
    await asyncio.to_thread(pipe.execute)

async def update_thread(cid, user_text, bot_text):
    key = await get_thread_key(cid)
    hist = await get_thread_history(cid)
    entry = f"\nUser: {user_text}\nBot: {bot_text}"
    new_h = (hist + entry)[-2000:]  # Keep last 2000 chars to prevent overflow
    
    pipe = redis_client.pipeline()
    pipe.hset(key, "history", new_h)
    pipe.expire(key, 86400)  # 24 hour TTL
    await asyncio.to_thread(pipe.execute)

async def ask_grok(prompt: str) -> str:
    """Ask Grok AI for a response based on the prompt."""
    # Implement exponential backoff for API requests
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            resp = await asyncio.to_thread(
                requests.post,
                GROK_URL,
                json={
                    "model": "grok-3-latest",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 180,
                    "temperature": 0.8
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to get Grok response after {max_retries} attempts: {e}")
                return "I'm feeling bullish on $DEGEN today! NFA."
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Grok API request failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

async def fetch_data(addr: str) -> Dict[str, Any]:
    """Fetch token data from DEX Screener."""
    try:
        resp = await asyncio.to_thread(
            requests.get,
            f"{DEXS_URL}{addr}", 
            timeout=10
        )
        resp.raise_for_status()
        j = resp.json()
        data = j[0] if isinstance(j, list) else j
        base = data.get("baseToken", {})
        
        return {
            "symbol": base.get("symbol", "DEGEN"),
            "price_usd": float(data.get("priceUsd", 0)),
            "volume_usd": float(data.get("volume", {}).get("h24", 0)),
            "market_cap": float(data.get("marketCap", 0)),
            "change_1h": float(data.get("priceChange", {}).get("h1", 0)),
            "change_24h": float(data.get("priceChange", {}).get("h24", 0)),
            "link": f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Error fetching data for {addr}: {e}")
        # Return fallback data to avoid crashing
        return {
            "symbol": "DEGEN",
            "price_usd": 0.0,
            "volume_usd": 0.0,
            "market_cap": 0.0,
            "change_1h": 0.0,
            "change_24h": 0.0,
            "link": f"https://dexscreener.com/solana/{addr}"
        }

def format_metrics(d: Dict[str, Any]) -> str:
    """Format token metrics for display in tweets."""
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%\n"
    )

async def lookup_address(token: str) -> Optional[str]:
    """Look up a token's contract address by symbol or return the address if already provided."""
    t = token.lstrip("$")
    if t.upper() == "DEGEN":
        return DEGEN_ADDR
    if ADDR_RE.fullmatch(t):
        return t
    
    try:
        resp = await asyncio.to_thread(
            requests.get,
            DEXS_SEARCH_URL + t,
            timeout=10
        )
        if resp.status_code != 200:
            logger.warning(f"Dex search for '{t}' returned {resp.status_code}")
            return None
            
        toks = resp.json().get("tokens", [])
        for item in toks:
            if item.get("symbol", "").lower() == t.lower():
                return item.get("contractAddress")
        return toks[0].get("contractAddress") if toks else None
    except Exception as e:
        logger.warning(f"Error looking up token '{t}': {e}")
        return None

# ——— RATE LIMITING & API SAFETY ———————————————————

async def safe_api_call(fn, queue: deque, limit: int, *args, **kwargs):
    """Execute API calls with rate limiting and exponential backoff."""
    now = time.time()
    
    # Clean expired timestamps from queue
    while queue and now - queue[0] > RATE_WINDOW:
        queue.popleft()
    
    # Wait if we're at the rate limit
    if len(queue) >= limit:
        wait = RATE_WINDOW - (now - queue[0]) + 1
        logger.info(f"Rate limit for {fn.__name__}, sleeping {wait:.1f}s")
        await asyncio.sleep(wait)
    
    # Make the API call with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(fn, *args, **kwargs)
            queue.append(time.time())
            return result
        except tweepy.TooManyRequests as e:
            # Handle Twitter rate limiting
            reset = int(e.response.headers.get("x-rate-limit-reset", time.time() + RATE_WINDOW))
            wait_time = reset - time.time() + 1
            logger.warning(f"Rate limited by Twitter API, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            if attempt == max_retries - 1:
                raise
        except tweepy.TwitterServerError as e:
            # Handle Twitter server errors with backoff
            if attempt == max_retries - 1:
                logger.error(f"Twitter API server error after {max_retries} attempts: {e}")
                raise
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Twitter API server error, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error in {fn.__name__}: {e}")
            if attempt == max_retries - 1:
                raise
            wait_time = retry_delay * (2 ** attempt)
            await asyncio.sleep(wait_time)

async def safe_mention_lookup(fn, **kw):
    return await safe_api_call(fn, mentions_q, MENTIONS_LIMIT, **kw)

async def safe_search(fn, **kw):
    return await safe_api_call(fn, search_q, SEARCH_LIMIT, **kw)

async def safe_tweet(text: str, media_id=None, **kw):
    def send(t, m, **kw2):
        return x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw2)
    return await safe_api_call(send, tweets_q, TWEETS_LIMIT, text, media_id, **kw)

async def upload_media(path: str) -> str:
    """Upload media to Twitter with retry logic."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            media = await asyncio.to_thread(x_api.media_upload, path)
            return media.media_id_string
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to upload media after {max_retries} attempts: {e}")
                raise
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Media upload failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

# ——— BOT LOGIC ——————————————————————————————————

async def post_raid(tweet):
    """Post a raid response to a tweet."""
    cid = tweet.conversation_id or tweet.id
    hist = await get_thread_history(cid)
    prompt = (f"History:{hist}\nUser: '{tweet.text}'\n"
              "Write a one-liner bullpost for $DEGEN… End with NFA.")
    msg = await ask_grok(prompt)
    
    # Get a random image with safe error handling
    try:
        img_files = glob.glob("raid_images/*.jpg")
        if not img_files:
            logger.warning("No raid images found")
            await safe_tweet(truncate_to_sentence(msg, 240), in_reply_to_tweet_id=tweet.id)
        else:
            img = random.choice(img_files)
            mid = await upload_media(img)
            await safe_tweet(truncate_to_sentence(msg, 240), media_id=mid, in_reply_to_tweet_id=tweet.id)
    except Exception as e:
        logger.error(f"Error with raid image: {e}")
        # Fall back to text-only tweet
        await safe_tweet(truncate_to_sentence(msg, 240), in_reply_to_tweet_id=tweet.id)
    
    # Mark as replied
    await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    """Handle a mention to the bot."""
    cid = tw.conversation_id or tw.id
    
    # Initialize thread history
    thread_count = await asyncio.to_thread(redis_client.hget, await get_thread_key(cid), "count")
    if thread_count is None:
        try:
            root_tweet = await asyncio.to_thread(
                x_client.get_tweet, 
                cid, 
                tweet_fields=["text"]
            )
            root = root_tweet.data.text
        except Exception as e:
            logger.warning(f"Couldn't get root tweet {cid}: {e}")
            root = "Unknown"
        await update_thread(cid, f"ROOT: {root}", "")

    # Clean up mention text by removing the bot's username
    txt = re.sub(rf"@{BOT_USERNAME}\b", "", tw.text, flags=re.IGNORECASE).strip()

    # 1) Handle raid command
    if re.search(r"\braid\b", txt, re.IGNORECASE):
        return await post_raid(tw)

    # 2) Handle CA / DEX / address commands
    if re.match(r"^\s*ca\s*$", txt, re.IGNORECASE):
        await safe_tweet(f"$DEGEN Contract Address: {DEGEN_ADDR}", in_reply_to_tweet_id=tw.id)
        await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return

    if re.match(r"^\s*dex\s*$", txt, re.IGNORECASE):
        metrics = format_metrics(await fetch_data(DEGEN_ADDR))
        await safe_tweet(metrics + f"\n{DEGEN_ADDR}", in_reply_to_tweet_id=tw.id)
        await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return

    if re.search(r"\b(contract|address)\b", txt, re.IGNORECASE) and not re.search(r"\bdex\b", txt, re.IGNORECASE):
        await safe_tweet(f"$DEGEN Contract Address: {DEGEN_ADDR}", in_reply_to_tweet_id=tw.id)
        await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return

    # 3) Handle token/address lookup
    token = next((w for w in txt.split() if SYMBOL_RE.match(w) or ADDR_RE.match(w)), None)
    if token:
        addr = DEGEN_ADDR if token.lstrip("$").upper() == "DEGEN" else await lookup_address(token)
        if addr:
            metrics = format_metrics(await fetch_data(addr))
            await safe_tweet(metrics + f"\n{addr}", in_reply_to_tweet_id=tw.id)
            await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
        else:
            logger.info(f"No address found for token '{token}', falling back to Grok.")

    # 4) General fallback - use Grok AI
    hist = await get_thread_history(cid)
    prompt = (f"History:{hist}\nUser asked: \"{txt}\"\n"
              "Answer concisely, then mention stacking $DEGEN in gambler style. End with NFA.")
    raw = await ask_grok(prompt)
    reply = raw
    
    # Add DEGEN information if missing
    if "$DEGEN" not in reply:
        reply += f"\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
    elif DEGEN_ADDR not in reply:
        reply += f"\n\nContract Address: {DEGEN_ADDR}"

    # Try to attach an image
    try:
        img_files = glob.glob("raid_images/*.jpg")
        if img_files:
            img = random.choice(img_files)
            media_id = await upload_media(img)
            await safe_tweet(reply, media_id=media_id, in_reply_to_tweet_id=tw.id)
        else:
            await safe_tweet(reply, in_reply_to_tweet_id=tw.id)
    except Exception as e:
        logger.error(f"Error with reply image: {e}")
        # Fall back to text-only tweet
        await safe_tweet(reply, in_reply_to_tweet_id=tw.id)

    # Update conversation history
    await update_thread(cid, txt, reply)
    await increment_thread(cid)
    await asyncio.to_thread(redis_client.sadd, f"{REDIS_PREFIX}replied_ids", str(tw.id))

# ——— LOOPS —————————————————————————————————————

async def mention_loop():
    """Monitor mentions and respond to them."""
    logger.info("Starting mention monitoring loop")
    
    error_backoff = 1
    
    while not shutdown_event.is_set():
        try:
            # Get the latest mention ID
            last = await asyncio.to_thread(redis_client.get, f"{REDIS_PREFIX}last_mention_id")
            
            # Prepare request parameters
            params = {
                "id": BOT_ID,
                "tweet_fields": ["id", "text", "conversation_id"],
                "max_results": 10
            }
            if last:
                params["since_id"] = int(last)
                
            # Fetch mentions
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            
            if res and res.data:
                for tw in reversed(res.data):
                    # Check if we've already replied
                    is_replied = await asyncio.to_thread(
                        redis_client.sismember,
                        f"{REDIS_PREFIX}replied_ids",
                        str(tw.id)
                    )
                    
                    if not is_replied:
                        # Update the last mention ID
                        await asyncio.to_thread(
                            redis_client.set,
                            f"{REDIS_PREFIX}last_mention_id",
                            tw.id
                        )
                        # Handle the mention
                        await handle_mention(tw)
            
            # Reset backoff on success
            error_backoff = 1
            
        except Exception as e:
            logger.exception(f"Mention loop error: {e}")
            # Exponential backoff on errors
            await asyncio.sleep(error_backoff * 10)
            error_backoff = min(error_backoff * 2, 60)  # Cap at 10 minutes
            
        # Regular polling interval
        await asyncio.sleep(110)

async def search_mentions_loop():
    """Search for mentions that might be missed by the mentions API."""
    logger.info("Starting search mentions loop")
    
    # Initialize last search ID if it doesn't exist
    exists = await asyncio.to_thread(redis_client.exists, f"{REDIS_PREFIX}last_search_id")
    if not exists:
        await asyncio.to_thread(redis_client.set, f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
    
    error_backoff = 1
    
    while not shutdown_event.is_set():
        try:
            # Search for recent mentions
            res = await safe_search(
                x_client.search_recent_tweets,
                query=f"@{BOT_USERNAME} -is:retweet",
                tweet_fields=["id", "text", "conversation_id"],
                max_results=10
            )
            
            if res and res.data:
                # Find the newest tweet ID
                newest = max(int(t.id) for t in res.data)
                
                for tw in res.data:
                    # Check if we've already replied
                    is_replied = await asyncio.to_thread(
                        redis_client.sismember, 
                        f"{REDIS_PREFIX}replied_ids", 
                        str(tw.id)
                    )
                    
                    if not is_replied:
                        await handle_mention(tw)
                
                # Update the last search ID
                await asyncio.to_thread(
                    redis_client.set,
                    f"{REDIS_PREFIX}last_search_id",
                    str(newest)
                )
            
            # Reset backoff on success
            error_backoff = 1
            
        except Exception as e:
            logger.exception(f"Search loop error: {e}")
            # Exponential backoff on errors
            await asyncio.sleep(error_backoff * 10)
            error_backoff = min(error_backoff * 2, 60)  # Cap at 10 minutes
            
        # Regular polling interval
        await asyncio.sleep(180)

async def hourly_post_loop():
    """Post hourly updates about the DEGEN token."""
    logger.info("Starting hourly post loop")
    
    error_backoff = 1
    
    while not shutdown_event.is_set():
        try:
            # Fetch token data
            data = await fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            
            # Get the bullish update from Grok
            raw = await ask_grok("Write a one-sentence bullpost update on $DEGEN. Be promotional.")
            tweet = truncate_to_sentence(metrics + raw, 560)
            
            # Check if this is different from the last post
            last = await asyncio.to_thread(redis_client.get, f"{REDIS_PREFIX}last_hourly_post")
            
            if tweet != last:
                # Try to attach an image
                try:
                    img_files = glob.glob("raid_images/*.jpg")
                    if img_files:
                        img = random.choice(img_files)
                        media_id = await upload_media(img)
                        await safe_tweet(tweet, media_id=media_id)
                    else:
                        await safe_tweet(tweet)
                except Exception as e:
                    logger.error(f"Error with hourly post image: {e}")
                    # Fall back to text-only tweet
                    await safe_tweet(tweet)
                
                # Update the last hourly post
                await asyncio.to_thread(redis_client.set, f"{REDIS_PREFIX}last_hourly_post", tweet)
            
            # Reset backoff on success
            error_backoff = 1
            
        except Exception as e:
            logger.exception(f"Hourly post error: {e}")
            # Exponential backoff on errors
            await asyncio.sleep(error_backoff * 60)
            error_backoff = min(error_backoff * 2, 60)  # Cap at 60 minutes
            
        # Regular hourly interval (adjusted for any processing time)
        await asyncio.sleep(3600)

# ——— RENDER SPECIFIC ——————————————————————————————

async def health_check_loop():
    """Simple health check for Render."""
    while not shutdown_event.is_set():
        logger.info("Health check: Bot is running")
        await asyncio.sleep(600)  # 10 minutes

def handle_sigterm():
    """Handle SIGTERM signal from Render."""
    logger.info("Received SIGTERM, initiating shutdown")
    shutdown_event.set()

# ——— MAIN —————————————————————————————————————

async def setup():
    """Setup global resources."""
    global redis_client, BOT_ID, BOT_USERNAME
    
    # Connect to Redis
    async with get_redis_connection() as r:
        redis_client = r
        
        # Get bot identity from twitter api
        try:
            me = await asyncio.to_thread(x_client.get_me)
            BOT_ID = me.data.id
            BOT_USERNAME = me.data.username
            logger.info(f"Authenticated as @{BOT_USERNAME} (ID: {BOT_ID})")
        except Exception as e:
            logger.error(f"Failed to get bot identity: {e}")
            raise
            
async def main():
    """Main entry point."""
    # Register signal handlers for Render
    signal.signal(signal.SIGTERM, lambda s, f: handle_sigterm())
    
    try:
        # Setup resources
        await setup()
        
        # Start all the loops
        tasks = [
            asyncio.create_task(mention_loop()),
            asyncio.create_task(search_mentions_loop()),
            asyncio.create_task(hourly_post_loop()),
            asyncio.create_task(health_check_loop())
        ]
        
        # Wait for all tasks to complete or shutdown signal
        _, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=None
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            
        # Wait for tasks to finish cancelling
        await asyncio.gather(*pending, return_exceptions=True)
        
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        # Exit gracefully but with error code
        return 1
    finally:
        logger.info("Bot is shutting down")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        exit(0)