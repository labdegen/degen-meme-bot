import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import json
import asyncio
import time
from collections import deque
from random import choice
import glob
import http.client
import sys
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of problematic tweet IDs to always skip
BLOCKED_TWEET_IDS = ["1924845778821845267"]

# Load environment variables
load_dotenv()
required = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Twitter API setup
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

try:
    me = x_client.get_me().data
    BOT_ID = me.id
    BOT_USERNAME = me.username
    logger.info(f"Authenticated as: {BOT_USERNAME} (ID: {BOT_ID})")
except Exception as e:
    logger.error(f"Authentication failed: {e}")
    exit(1)

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
redis_client.ping()
logger.info("Redis connected")

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

ADDR_RE = re.compile(r"\b[A-Za-z0-9]{43,44}\b")
SYMBOL_RE = re.compile(r"\$([A-Za-z0-9]{2,10})", re.IGNORECASE)
USERNAME_RE = re.compile(rf"@{BOT_USERNAME}\b", re.IGNORECASE)

RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
SEARCH_LIMIT = 10
mentions_timestamps = deque()
tweet_timestamps = deque()
search_timestamps = deque()

# Set initial search ID to current time-based ID to avoid the "since_id too old" error
current_time_ms = int(time.time() * 1000) - 1728000000
INITIAL_SEARCH_ID = str((current_time_ms << 22))

# Helpers
def truncate_to_sentence(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    snippet = text[:max_length]
    for sep in ('. ', '! ', '? '):
        idx = snippet.rfind(sep)
        if idx != -1:
            return snippet[:idx+1]
    return snippet

def clean_reply_text(text: str) -> str:
    """Clean reply text properly - remove @mentions and fix punctuation"""
    # Remove @username patterns
    cleaned = re.sub(r'@\w+', '', text)
    
    # Fix common punctuation issues from @mention removal
    cleaned = re.sub(r'^[\s,\.\!]+', '', cleaned)  # Remove leading punctuation
    cleaned = re.sub(r'[\s,]+([,\.])', r'\1', cleaned)  # Fix double punctuation  
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Clean multiple spaces
    cleaned = cleaned.strip()
    
    # Ensure it starts with a capital letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned

def fix_degen_spacing(text: str) -> str:
    """Ensure $DEGEN has proper spacing for cashtag to work"""
    text = re.sub(r'([^\s])(\$DEGEN)', r'\1 \2', text)  # Add space before if missing
    text = re.sub(r'(\$DEGEN)([^\s\.\,\!\?])', r'\1 \2', text)  # Add space after if missing  
    text = re.sub(r'-\s*\$DEGEN', r' $DEGEN', text)  # Fix "-$DEGEN" to " $DEGEN"
    return text

# Thread memory helpers
def get_thread_key(cid):
    return f"{REDIS_PREFIX}thread:{cid}"

def get_thread_history(cid):
    return redis_client.hget(get_thread_key(cid), "history") or ""

def increment_thread(cid):
    redis_client.hincrby(get_thread_key(cid), "count", 1)
    redis_client.expire(get_thread_key(cid), 86400)

def update_thread(cid, user_text, bot_text):
    hist = get_thread_history(cid)
    entry = f"\nUser: {user_text}\nBot: {bot_text}"
    new_hist = (hist + entry)[-2000:]
    redis_client.hset(get_thread_key(cid), "history", new_hist)
    redis_client.expire(get_thread_key(cid), 86400)

# Grok prompt
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. No slang. High class but a little edgy like Don Draper. "
    "IF DEGEN, or $DEGEN is mentioned, always ONLY speak about the $DEGEN token at contract address {DEGEN_ADDR}. DO not include the contract address in the reply. "
    "Do NOT mention any other token or chain when it comes to DEGEN. Other tokens you can reply honestly."
)

def ask_grok(prompt: str) -> str:
    payload = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {os.getenv('GROK_API_KEY')}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROK_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}")
        return "Unable to provide an update at this time."

async def safe_api_call(fn, timestamps_queue, limit, *args, **kwargs):
    now = time.time()
    while timestamps_queue and now - timestamps_queue[0] > RATE_WINDOW:
        timestamps_queue.popleft()
    if len(timestamps_queue) >= limit:
        await asyncio.sleep(RATE_WINDOW - (now - timestamps_queue[0]) + 1)
    try:
        result = fn(*args, **kwargs)
        timestamps_queue.append(time.time())
        return result
    except (requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
        logger.warning(f"Network error during API call: {e}. Retrying in 5s…")
        await asyncio.sleep(5)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(reset - time.time() + 1)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.BadRequest as e:
        raise e
    except Exception as e:
        logger.error(f"API call error: {e}", exc_info=True)
        raise e

async def safe_mention_lookup(fn, *args, **kwargs):
    return await safe_api_call(fn, mentions_timestamps, MENTIONS_LIMIT, *args, **kwargs)

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, search_timestamps, SEARCH_LIMIT, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, **kwargs):
    # CLEAN ALL @MENTIONS FROM TEXT AND FIX $DEGEN SPACING
    cleaned_text = clean_reply_text(text)
    cleaned_text = fix_degen_spacing(cleaned_text)
    
    return await safe_api_call(
        lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
        tweet_timestamps, 
        TWEETS_LIMIT,
        cleaned_text, 
        media_id, 
        **kwargs
    )

# DEX helpers
def fetch_data(addr: str) -> dict:
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()[0] if isinstance(r.json(), list) else r.json()
        base = data.get('baseToken', {})
        return {
            'symbol': base.get('symbol','DEGEN'),
            'price_usd': float(data.get('priceUsd',0)),
            'volume_usd': float(data.get('volume',{}).get('h24',0)),
            'market_cap': float(data.get('marketCap',0)),
            'change_1h': float(data.get('priceChange',{}).get('h1',0)),
            'change_24h': float(data.get('priceChange',{}).get('h24',0)),
            'link': f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(d: dict) -> str:
    """Format DEX data with icons - EXACTLY as your original working format"""
    if not d:
        return "🚀 DEGEN | Data loading...\nMC Loading... | Vol24 Loading...\n1h Loading... | 24h Loading..."
    
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"
    )

def lookup_address(token: str) -> str:
    t = token.lstrip('$')
    if t.upper() == 'DEGEN': return DEGEN_ADDR
    if ADDR_RE.fullmatch(t): return t
    try:
        r = requests.get(DEXS_SEARCH_URL + t, timeout=10)
        r.raise_for_status()
        toks = r.json().get('tokens', [])
        for item in toks:
            if item.get('symbol','').lower() == t.lower():
                return item.get('contractAddress')
        if toks: return toks[0].get('contractAddress')
    except:
        pass
    return None

def build_dex_reply(addr: str) -> str:
    """Build DEX reply - ORIGINAL WORKING FORMAT"""
    data = fetch_data(addr)
    if not data:
        return f"Data temporarily unavailable\n\nhttps://dexscreener.com/solana/{addr}"
    
    return format_metrics(data) + "\n\n" + data['link']

async def post_raid(tweet):
    """FIXED RAID FUNCTION - Always includes $DEGEN and contract address"""
    try:
        convo_id = tweet.conversation_id or tweet.id
        history = get_thread_history(convo_id)
        
        prompt = (
            f"History:{history}\n"
            f"User: '{tweet.text}'\n"
            "Write a one-liner bullpost for $DEGEN based on the above. "
            f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA. No slang. High class but a little edgy like Don Draper."
        )
        
        msg = ask_grok(prompt)
        
        # FORCE $DEGEN and contract address to be included
        if "$DEGEN" not in msg:
            msg = f"Interesting perspective! Have you looked into $DEGEN? {msg}"
        
        # Clean any existing contract address and add cleanly at end
        msg = msg.replace(f"CA: {DEGEN_ADDR}", "").replace(f"ca: {DEGEN_ADDR}", "").strip()
        msg = msg.replace(DEGEN_ADDR, "").strip()
        
        # ALWAYS add contract address at the end
        msg = f"{msg}\n\nCA: {DEGEN_ADDR}"
        
        # Try to add meme image
        media_id = None
        try:
            meme_files = glob.glob("raid_images/*.jpg")
            if meme_files:
                img = choice(meme_files)
                media_id = x_api.media_upload(img).media_id_string
        except Exception as e:
            logger.warning(f"Meme upload failed: {e}")
        
        await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            media_id=media_id,
            in_reply_to_tweet_id=tweet.id
        )
        
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
        logger.info(f"✅ Posted raid reply with $DEGEN and CA to tweet {tweet.id}")
        
    except Exception as e:
        logger.error(f"Error in post_raid for tweet {tweet.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def search_degen_loop():
    """Search for 'degen' mentions and raid them"""
    key = f"{REDIS_PREFIX}last_degen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)

    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "degen -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    tid = str(tw.id)
                    if tid in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", tid):
                        continue
                    await post_raid(tw)
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tid)
                redis_client.set(key, str(newest))
                logger.info(f"🎯 Processed {len(res.data)} degen mentions")
        except Exception as e:
            logger.error(f"search_degen_loop error: {e}", exc_info=True)
        await asyncio.sleep(180)  # every 3 minutes

async def handle_mention(tw):
    """FIXED MENTION HANDLING - Single function, clean replies"""
    try:
        convo_id = tw.conversation_id or tw.id
        if redis_client.hget(get_thread_key(convo_id), "count") is None:
            try:
                root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
                update_thread(convo_id, f"ROOT: {root}", "")
            except Exception as e:
                logger.warning(f"Failed to get root tweet: {e}")
                update_thread(convo_id, f"ROOT: Unknown", "")
        
        history = get_thread_history(convo_id)
        txt = re.sub(rf"@{BOT_USERNAME}\b", "", tw.text, flags=re.IGNORECASE).strip()

        # 1) raid command
        if re.search(r"\braid\b", txt, re.IGNORECASE):
            await post_raid(tw)
            return

        # 2) Check for CA command (contract address only)
        if re.search(r"\bca\b", txt, re.IGNORECASE) and not re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
                in_reply_to_tweet_id=tw.id
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 3) Check for DEX or other contract address commands
        if re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=build_dex_reply(DEGEN_ADDR),
                in_reply_to_tweet_id=tw.id
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 4) token/address -> DEX preview
        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
        if token:
            sym = token.lstrip('$').upper()
            addr = DEGEN_ADDR if sym=="DEGEN" else lookup_address(token)
            if addr:
                await safe_tweet(
                    text=build_dex_reply(addr),
                    in_reply_to_tweet_id=tw.id
                )
                redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                return

        # 5) general fallback - ALWAYS include $DEGEN and contract address
        prompt = (
            f"History:{history}\n"
            f"User asked: \"{txt}\"\n"
            "Answer naturally and concisely."
        )
        raw = ask_grok(prompt)
        
        reply_body = raw.strip()
        
        # ALWAYS include $DEGEN and contract address
        if "$DEGEN" not in reply_body:
            reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
        else:
            if DEGEN_ADDR not in reply_body:
                reply = f"{reply_body}\n\nStack $DEGEN. CA: {DEGEN_ADDR}"
            else:
                reply = reply_body
        
        if len(reply) > 360:
            reply = truncate_to_sentence(reply, 360) + f"\n\n$DEGEN. CA: {DEGEN_ADDR}"
        
        # Try to use meme image
        media_id = None
        try:
            meme_files = glob.glob("raid_images/*.jpg")
            if meme_files:
                img = choice(meme_files)
                media_id = x_api.media_upload(img).media_id_string
        except Exception as e:
            logger.warning(f"Meme upload failed: {e}")
        
        await safe_tweet(
            text=reply,
            media_id=media_id,
            in_reply_to_tweet_id=tw.id
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        update_thread(convo_id, txt, reply)
        increment_thread(convo_id)
        
    except Exception as e:
        logger.error(f"Error handling mention {tw.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

async def search_mentions_loop():
    """Search for mentions that might not be captured by the mentions API"""
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
        logger.info(f"Initialized last_search_id to {INITIAL_SEARCH_ID}")
    
    while True:
        try:
            query = f"@{BOT_USERNAME} -is:retweet"
            search_params = {
                "query": query,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            
            try:
                search_results = await safe_search(
                    x_client.search_recent_tweets,
                    **search_params
                )
                logger.info("Successfully searched for community mentions")
            except Exception as e:
                logger.error(f"Search call failed: {e}", exc_info=True)
                search_results = None
            
            if search_results and search_results.data:
                newest_id = max(int(tw.id) for tw in search_results.data) if search_results.data else 0
                
                for tw in search_results.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS:
                        logger.info(f"Skipping blocked tweet ID: {tw.id}")
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        continue
                        
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    try:
                        logger.info(f"Processing community mention: {tw.id} - {tw.text[:30]}...")
                        await handle_mention(tw)
                    except Exception as e:
                        logger.error(f"Error processing mention {tw.id}: {e}", exc_info=True)
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                if newest_id > 0:
                    redis_client.set(f"{REDIS_PREFIX}last_search_id", str(newest_id))
                    logger.info(f"Updated last_search_id to {newest_id}")
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        await asyncio.sleep(180)  # every 3 minutes

async def hourly_post_loop():
    """FIXED HOURLY POSTS - Exact format from your original working code"""
    grok_prompts = [
        "Write a positive one-sentence analytical update on $DEGEN using data from the last hour. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace.",
        "Write a positive one-sentence cryptic message about secret tech being developed on $DEGEN's price action. Be edgy and risky. Do not mention the contract address. No slang. High class but a little edgy like Don Draper.",
        "Write a one sentence, cryptic message about $DEGEN that implies insider knowledge. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace.",
        "Write a one sentence, cryptic comment about people who haven't bought $DEGEN yet. Do not mention the contract address. No slang. High class but a little edgy like Elon Musk.",
        "Write a one sentence comparing $DEGEN to the broader crypto market. Be cryptic. Do not mention the contract address. No slang. High class but a little edgy like Hemingway.",
        "Write a one sentence post about diamond hands and $DEGEN's future potential. Do not mention the contract address. No slang. High class but a little edgy like Hunter Thompson."
    ]
    
    hour_counter = 0

    while True:
        try:
            # Fetch on-chain and market data
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            dex_link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")

            # Ask Grok for a clean one-liner
            selected_prompt = grok_prompts[hour_counter % len(grok_prompts)]
            raw = ask_grok(selected_prompt).strip()

            # Build tweet: ORIGINAL WORKING FORMAT
            # metrics block, blank line, one-liner, blank line, link
            tweet = (
                metrics +
                "\n\n" +
                raw +
                "\n\n" +
                dex_link
            )

            # Only post if it's new
            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                await safe_tweet(tweet)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
                logger.info("✅ Posted hourly update")

            hour_counter += 1
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        
        await asyncio.sleep(3600)

async def main():
    try:
        logger.info("🚀 Starting FIXED CRYPTO PROMOTION bot for $DEGEN...")
        logger.info("✅ Fixed: Raid replies include $DEGEN+CA, proper hourly format, NO auto liking")
        
        # Pre-mark all blocked tweets as replied to
        for tweet_id in BLOCKED_TWEET_IDS:
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tweet_id)
            logger.info(f"Pre-marked blocked tweet ID {tweet_id} as replied")
        
        logger.info("💎 Starting bot functions (NO AUTO LIKING)...")
        
        # Run ONLY the essential loops - NO AUTO LIKING
        await asyncio.gather(
            search_mentions_loop(),      # Handle @mentions with $DEGEN+CA
            hourly_post_loop(),         # Fixed hourly format 
            search_degen_loop(),        # Search 'degen' and raid with $DEGEN+CA
        )
        
    except Exception as e:
        logger.error(f"Main function error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)