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

like_timestamps = deque()
LIKE_LIMIT = 50

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

# Optional environment variables
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
MINIMUM_BUY_SOL = 1.0

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
me = x_client.get_me().data
BOT_ID = me.id
BOT_USERNAME = me.username
logger.info(f"Authenticated as: {BOT_USERNAME} (ID: {BOT_ID})")

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
SEARCH_LIMIT = 15
LIKE_LIMIT = 50 
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
        return fn(*args, **kwargs)
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
    finally:
        timestamps_queue.append(time.time())

async def safe_mention_lookup(fn, *args, **kwargs):
    return await safe_api_call(fn, mentions_timestamps, MENTIONS_LIMIT, *args, **kwargs)

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, search_timestamps, SEARCH_LIMIT, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, **kwargs):
    return await safe_api_call(
        lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
        tweet_timestamps, 
        TWEETS_LIMIT,
        text, 
        media_id, 
        **kwargs
    )

async def safe_like(tweet_id: str):
    return await safe_api_call(
        lambda tid: x_api.create_favorite(id=tid),
        like_timestamps,
        LIKE_LIMIT,
        tweet_id
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
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%\n"
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
    data = fetch_data(addr)
    return format_metrics(data) + data['link']

async def post_raid(tweet):
    """Post a bullpost for $DEGEN based on the tweet"""
    try:
        convo_id = tweet.conversation_id or tweet.id
        history = get_thread_history(convo_id)
        
        # Get author info if available
        author_info = ""
        if hasattr(tweet, 'author_id'):
            try:
                user_info = x_client.get_user(id=tweet.author_id)
                if user_info and user_info.data:
                    author_info = f" (from @{user_info.data.username})"
            except:
                pass
        
        prompt = (
            f"History:{history}\n"
            f"User{author_info}: '{tweet.text}'\n"
            "Write a one-liner bullpost for $DEGEN based on the above. "
            f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. "
            "End with NFA. No slang. High class but a little edgy like Don Draper."
        )
        
        msg = ask_grok(prompt)
        
        # Ensure we have meme images available
        meme_files = glob.glob("raid_images/*.jpg")
        if not meme_files:
            logger.warning("No meme images found in raid_images/ directory")
            await safe_tweet(
                text=truncate_to_sentence(msg, 240),
                in_reply_to_tweet_id=tweet.id
            )
        else:
            img = choice(meme_files)
            media_id = x_api.media_upload(img).media_id_string
            await safe_tweet(
                text=truncate_to_sentence(msg, 240),
                media_id=media_id,
                in_reply_to_tweet_id=tweet.id
            )
        
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
        logger.info(f"Successfully posted raid reply to tweet {tweet.id}")
        
    except Exception as e:
        logger.error(f"Error in post_raid for tweet {tweet.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def search_degen_loop():
    """Enhanced search for 'degen' tweets from users with 500+ followers"""
    key = f"{REDIS_PREFIX}last_degen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)

    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "degen -is:retweet -is:reply",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "conversation_id", "created_at", "author_id"],
                "expansions": ["author_id"],
                "user_fields": ["username", "public_metrics"],
                "max_results": 50
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                
                # Create a mapping of user_id to user data for follower filtering
                user_map = {}
                if hasattr(res, 'includes') and res.includes and 'users' in res.includes:
                    for user in res.includes['users']:
                        user_map[user.id] = user
                
                qualified_tweets = []
                for tw in res.data:
                    tid = str(tw.id)
                    
                    # Skip blocked tweets and already replied tweets
                    if tid in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", tid):
                        continue
                    
                    # Check follower count
                    author = user_map.get(tw.author_id)
                    if author and hasattr(author, 'public_metrics'):
                        follower_count = author.public_metrics.get('followers_count', 0)
                        if follower_count >= 500:
                            qualified_tweets.append(tw)
                            logger.info(f"Found qualified tweet from @{author.username} ({follower_count} followers): {tw.text[:50]}...")
                
                # Process qualified tweets (limit to avoid rate limits)
                for tw in qualified_tweets[:3]:
                    try:
                        await post_raid(tw)
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        await asyncio.sleep(10)
                    except Exception as e:
                        logger.error(f"Error processing qualified tweet {tw.id}: {e}")
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                redis_client.set(key, str(newest))
                logger.info(f"Processed {len(qualified_tweets)} qualified tweets out of {len(res.data)} total")
                
        except Exception as e:
            logger.error(f"search_degen_loop error: {e}", exc_info=True)
        
        await asyncio.sleep(240)

async def auto_like_degen_loop():
    """Enhanced auto-like for 'degen' tweets from users with decent following"""
    key = f"{REDIS_PREFIX}last_like_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)

    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "degen -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "author_id"],
                "expansions": ["author_id"],
                "user_fields": ["username", "public_metrics"],
                "max_results": 50
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                
                # Create user mapping for follower filtering
                user_map = {}
                if hasattr(res, 'includes') and res.includes and 'users' in res.includes:
                    for user in res.includes['users']:
                        user_map[user.id] = user
                
                liked_count = 0
                for tw in res.data:
                    tid = str(tw.id)
                    
                    if redis_client.sismember(f"{REDIS_PREFIX}liked_ids", tid):
                        continue
                    
                    tweet_text_upper = tw.text.upper()
                    if "DEGEN" in tweet_text_upper:
                        author = user_map.get(tw.author_id)
                        follower_count = 0
                        if author and hasattr(author, 'public_metrics'):
                            follower_count = author.public_metrics.get('followers_count', 0)
                        
                        if follower_count >= 100:
                            try:
                                await safe_like(tid)
                                redis_client.sadd(f"{REDIS_PREFIX}liked_ids", tid)
                                liked_count += 1
                                logger.info(f"Liked tweet from @{author.username if author else 'unknown'} ({follower_count} followers)")
                                
                                if liked_count >= 10:
                                    break
                                    
                                await asyncio.sleep(2)
                            except Exception as e:
                                logger.error(f"Error liking tweet {tid}: {e}")
                                redis_client.sadd(f"{REDIS_PREFIX}liked_ids", tid)
                
                redis_client.set(key, str(newest))
                logger.info(f"Liked {liked_count} tweets this cycle")
                
        except Exception as e:
            logger.error(f"auto_like_degen_loop error: {e}", exc_info=True)
        
        await asyncio.sleep(360)

async def handle_mention(tw):
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

        # 1) raid
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
                text=f"{build_dex_reply(DEGEN_ADDR)}",
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

        # 5) general fallback
        prompt = (
            f"History:{history}\n"
            f"User asked: \"{txt}\"\n"
            "First, answer naturally and concisely."
        )
        raw = ask_grok(prompt)
        
        reply_body = raw.strip()
        
        if "$DEGEN" not in reply_body:
            reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
        else:
            if DEGEN_ADDR not in reply_body:
                reply = f"{reply_body}\n\nStack $DEGEN. ca: {DEGEN_ADDR}"
            else:
                reply = reply_body
        
        if len(reply) > 360:
            reply = truncate_to_sentence(reply, 360) + f"\n\n$DEGEN. ca: {DEGEN_ADDR}"
        
        meme_files = glob.glob("raid_images/*.jpg")
        if meme_files:
            img = choice(meme_files)
            media_id = x_api.media_upload(img).media_id_string
        else:
            media_id = None
        
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
        
        await asyncio.sleep(180)

async def hourly_post_loop():
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
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            dex_link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")

            selected_prompt = grok_prompts[hour_counter % len(grok_prompts)]
            raw = ask_grok(selected_prompt).strip()

            tweet = (
                metrics.rstrip() +
                "\n\n" +
                raw +
                "\n\n" +
                dex_link
            )

            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                await safe_tweet(tweet)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
                logger.info("Posted hourly update")

            hour_counter += 1
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        
        await asyncio.sleep(3600)

async def monitor_volume_spikes_loop():
    """Monitor for volume spikes that indicate major buys"""
    previous_volume = None
    
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            current_volume = data.get('volume_usd', 0)
            current_change_1h = data.get('change_1h', 0)
            
            if previous_volume is not None and current_volume > 0:
                volume_increase = current_volume - previous_volume
                volume_increase_pct = (volume_increase / previous_volume) * 100 if previous_volume > 0 else 0
                
                # Detect significant volume spike (>50% increase in 5 minutes)
                if volume_increase_pct > 50 and volume_increase > 1000:
                    estimated_sol_value = volume_increase / 140
                    
                    if estimated_sol_value >= MINIMUM_BUY_SOL:
                        await post_volume_spike_tweet(estimated_sol_value, volume_increase, current_change_1h)
                        logger.info(f"Posted volume spike tweet: ${volume_increase:,.0f} volume increase")
                
                # Detect significant price pump (>20% in 1 hour)
                elif current_change_1h > 20:
                    estimated_buy_value = current_volume * 0.1
                    estimated_sol = estimated_buy_value / 140
                    
                    if estimated_sol >= MINIMUM_BUY_SOL:
                        await post_price_pump_tweet(current_change_1h, estimated_sol)
                        logger.info(f"Posted price pump tweet: {current_change_1h:.1f}% pump")
            
            previous_volume = current_volume
            
        except Exception as e:
            logger.error(f"Volume monitoring error: {e}")
        
        await asyncio.sleep(300)  # Check every 5 minutes

async def post_volume_spike_tweet(estimated_sol, volume_increase, price_change):
    """Post tweet about volume spike indicating major buy"""
    try:
        # Check cooldown
        if redis_client.exists(f"{REDIS_PREFIX}volume_spike_posted"):
            return
            
        spike_prompts = [
            f"🚨 VOLUME SPIKE! ${volume_increase:,.0f} just flowed into $DEGEN (~{estimated_sol:.1f} SOL)! Someone's loading up! 💎",
            f"🐋 BIG MONEY MOVING! ${volume_increase:,.0f} volume spike on $DEGEN! Smart money is accumulating! 🚀",
            f"💰 WHALE ACTIVITY! Massive ${volume_increase:,.0f} buy pressure on $DEGEN! Don't sleep on this! 👀",
            f"🔥 VOLUME EXPLOSION! ${volume_increase:,.0f} just hit $DEGEN! The smart money knows something! 💯"
        ]
        
        tweet_text = choice(spike_prompts)
        
        if price_change > 0:
            tweet_text += f"\n\nPrice pumping {price_change:+.1f}% in the last hour!"
        
        tweet_text += f"\n\nChart: https://dexscreener.com/solana/{DEGEN_ADDR}"
        
        meme_files = glob.glob("raid_images/*.jpg")
        if meme_files:
            img = choice(meme_files)
            media_id = x_api.media_upload(img).media_id_string
        else:
            media_id = None
        
        await safe_tweet(
            text=tweet_text,
            media_id=media_id
        )
        
        redis_client.setex(f"{REDIS_PREFIX}volume_spike_posted", 1800, "1")  # 30 min cooldown
        logger.info(f"Successfully posted volume spike tweet")
        
    except Exception as e:
        logger.error(f"Error posting volume spike tweet: {e}")

async def post_price_pump_tweet(price_change, estimated_sol):
    """Post tweet about significant price pump"""
    try:
        if redis_client.exists(f"{REDIS_PREFIX}pump_tweet_posted"):
            return
            
        pump_prompts = [
            f"🚀 $DEGEN PUMPING! Up {price_change:+.1f}% in the last hour! Someone knows something! 💎",
            f"📈 PRICE ALERT! $DEGEN surging {price_change:+.1f}%! Big money is moving! 🐋",
            f"🔥 $DEGEN ON FIRE! {price_change:+.1f}% pump! Don't get left behind! 💯",
            f"💰 BREAKOUT! $DEGEN up {price_change:+.1f}%! Smart money is accumulating! 🚀"
        ]
        
        tweet_text = choice(pump_prompts)
        tweet_text += f"\n\nLive chart: https://dexscreener.com/solana/{DEGEN_ADDR}"
        
        meme_files = glob.glob("raid_images/*.jpg")
        if meme_files:
            img = choice(meme_files)
            media_id = x_api.media_upload(img).media_id_string
        else:
            media_id = None
        
        await safe_tweet(
            text=tweet_text,
            media_id=media_id
        )
        
        redis_client.setex(f"{REDIS_PREFIX}pump_tweet_posted", 3600, "1")  # 1 hour cooldown
        logger.info(f"Successfully posted pump tweet: {price_change:.1f}%")
        
    except Exception as e:
        logger.error(f"Error posting pump tweet: {e}")

async def main():
    try:
        logger.info("🤖 Starting DEGEN bot...")
        
        # Pre-mark all blocked tweets as replied to
        for tweet_id in BLOCKED_TWEET_IDS:
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tweet_id)
            logger.info(f"Pre-marked blocked tweet ID {tweet_id} as replied")
        
        logger.info("✅ Bot initialized - starting all loops...")
        
        # Run all loops
        await asyncio.gather(
            search_mentions_loop(),
            hourly_post_loop(),
            search_degen_loop(),
            auto_like_degen_loop(),
            monitor_volume_spikes_loop(),  # This replaces WebSocket for now
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