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
from random import choice, randint
import glob
import http.client
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of problematic tweet IDs to always skip
BLOCKED_TWEET_IDS = ["1924845778821845267", "1926657606195593300", "1926648154012741852"]

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

# Twitter API setup - Keep v1.1 for media upload only
oauth = tweepy.OAuth1UserHandler(
    os.getenv("X_API_KEY"),
    os.getenv("X_API_KEY_SECRET"),
    os.getenv("X_ACCESS_TOKEN"),
    os.getenv("X_ACCESS_TOKEN_SECRET")
)
x_api = tweepy.API(oauth)  # Only for media upload

# Use v2 client for everything else
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

# Set initial search ID
current_time_ms = int(time.time() * 1000) - 1728000000
INITIAL_SEARCH_ID = str((current_time_ms << 22))

# Updated daily limits for Basic tier (from official X API docs)
DAILY_TWEET_LIMITS = {
    'main_posts': 18,        # Every 6 hours = 4 per day
    'crypto_bullposts': 12, # Conservative for 100/day limit  
    'mentions': 50,         # Most tweets should be mention replies
    'likes': 200,          # Official limit: 200/24hrs
    'retweets': 80,        # Calculated: 5 per 15min * 96 periods = 480, but be conservative
    'follows': 80,         # Calculated: 5 per 15min * 96 periods = 480, but be conservative
    'total_tweets': 100    # Official limit: 100 tweets/24hrs PER USER
}

# 15-minute rate limits for actions (from official X API docs)
FIFTEEN_MIN_LIMITS = {
    'retweets': 5,         # Official: 5 per 15min
    'follows': 5,          # Official: 5 per 15min  
    'searches': 50,        # Official: 60 per 15min, but be conservative
    'mentions_search': 10, # Official: 10 per 15min for mentions endpoint
    'tweets': 4            # Conservative: spread 100 daily tweets across day (4 per 15min max)
}

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

def get_daily_count(action_type):
    """Get current daily count for an action type"""
    key = f"{REDIS_PREFIX}daily:{action_type}:{time.strftime('%Y-%m-%d')}"
    count = redis_client.get(key)
    return int(count) if count else 0

def increment_daily_count(action_type):
    """Increment daily count for an action type"""
    key = f"{REDIS_PREFIX}daily:{action_type}:{time.strftime('%Y-%m-%d')}"
    redis_client.incr(key)
    redis_client.expire(key, 86400)  # Expire after 24 hours
    return get_daily_count(action_type)

def get_15min_count(action_type):
    """Get current 15-minute count for an action type"""
    now = datetime.now()
    # Round down to nearest 15-minute window
    window_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    key = f"{REDIS_PREFIX}15min:{action_type}:{window_start.strftime('%Y-%m-%d-%H-%M')}"
    count = redis_client.get(key)
    return int(count) if count else 0

def increment_15min_count(action_type):
    """Increment 15-minute count for an action type"""
    now = datetime.now()
    # Round down to nearest 15-minute window
    window_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    key = f"{REDIS_PREFIX}15min:{action_type}:{window_start.strftime('%Y-%m-%d-%H-%M')}"
    redis_client.incr(key)
    redis_client.expire(key, 900)  # Expire after 15 minutes
    return get_15min_count(action_type)

def can_perform_action(action_type):
    """Check if we can perform an action without hitting daily limits"""
    current_count = get_daily_count(action_type)
    limit = DAILY_TWEET_LIMITS.get(action_type, 0)
    return current_count < limit

def can_perform_15min_action(action_type):
    """Check if we can perform an action without hitting 15-minute limits"""
    current_count = get_15min_count(action_type)
    limit = FIFTEEN_MIN_LIMITS.get(action_type, 0)
    return current_count < limit

def can_post_tweet():
    """Check if we can post a tweet without hitting total daily limit"""
    total_tweets = (get_daily_count('main_posts') + 
                   get_daily_count('crypto_bullposts') + 
                   get_daily_count('mentions'))
    return total_tweets < DAILY_TWEET_LIMITS['total_tweets']

def contains_degen_contract(text: str) -> bool:
    """Check if tweet contains DEGEN contract address"""
    return DEGEN_ADDR in text

# Global flag to track daily tweet limit exhaustion
daily_tweet_limit_exhausted = False
daily_limit_reset_time = None

def check_daily_limit_exhaustion():
    """Check if daily limit reset time has passed"""
    global daily_tweet_limit_exhausted, daily_limit_reset_time
    
    if daily_limit_reset_time and time.time() > daily_limit_reset_time:
        daily_tweet_limit_exhausted = False
        daily_limit_reset_time = None
        logger.info("Daily tweet limit has reset - resuming tweet attempts")

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
    logger.info(f"safe_api_call: Making API call")
    
    try:
        logger.info("Making API call...")
        result = fn(*args, **kwargs)
        logger.info("API call successful")
        return result
    except (requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
        logger.warning(f"Network error during API call: {e}. Retrying in 5s…")
        await asyncio.sleep(5)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.TooManyRequests as e:
        global daily_tweet_limit_exhausted, daily_limit_reset_time
        
        logger.warning(f"Rate limit response: {e.response.text}")
        logger.warning(f"Rate limit headers: {e.response.headers}")
        
        # Check if it's daily user tweet limit exhaustion
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
            remaining = e.response.headers.get('x-user-limit-24hour-remaining', '1')
            reset_time_header = e.response.headers.get('x-user-limit-24hour-reset')
            
            if remaining == '0' and reset_time_header:
                reset_time = int(reset_time_header)
                daily_tweet_limit_exhausted = True
                daily_limit_reset_time = reset_time
                hours_until_reset = (reset_time - int(time.time())) / 3600
                
                logger.warning(f"🚫 DAILY TWEET LIMIT EXHAUSTED (100/100 used)")
                logger.warning(f"⏰ Resets in {hours_until_reset:.1f} hours at timestamp {reset_time}")
                logger.warning(f"🔄 Bot will continue likes/retweets/follows but skip ALL tweets until reset")
                
                return None  # Return None instead of sleeping
        
        # Try to parse regular rate limit reset from headers
        reset_time = None
        reset_header = e.response.headers.get('x-rate-limit-reset')
        if reset_header:
            try:
                reset_time = int(reset_header)
                sleep_duration = max(reset_time - int(time.time()) + 10, 60)  # Add 10s buffer
                logger.warning(f"15-min rate limit hit, sleeping for {sleep_duration}s")
                await asyncio.sleep(sleep_duration)
                return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
            except (ValueError, TypeError):
                pass
        
        # Fallback: sleep for 15 minutes
        logger.warning("Sleeping for 15 minutes due to rate limit")
        await asyncio.sleep(900)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.BadRequest as e:
        logger.error(f"BadRequest error: {e}")
        raise e
    except Exception as e:
        logger.error(f"API call error: {e}", exc_info=True)
        raise e

async def safe_search(fn, search_type='general', *args, **kwargs):
    # Different limits for different search types
    if search_type == 'mentions':
        if not can_perform_15min_action('mentions_search'):
            logger.warning(f"15-minute mentions search limit reached: {get_15min_count('mentions_search')}")
            return None
    else:
        if not can_perform_15min_action('searches'):
            logger.warning(f"15-minute search limit reached: {get_15min_count('searches')}")
            return None
    
    result = await safe_api_call(fn, None, 0, *args, **kwargs)
    if result:
        if search_type == 'mentions':
            increment_15min_count('mentions_search')
            logger.info(f"Mentions search completed - 15min count: {get_15min_count('mentions_search')}")
        else:
            increment_15min_count('searches')
            logger.info(f"Search completed - 15min count: {get_15min_count('searches')}")
    return result

async def safe_tweet(text: str, media_id=None, action_type='mentions', **kwargs):
    global daily_tweet_limit_exhausted
    
    # Check if daily limit is exhausted
    check_daily_limit_exhaustion()
    
    if daily_tweet_limit_exhausted:
        logger.info(f"⏭️ Skipping tweet ({action_type}) - daily limit exhausted until reset")
        return None
    
    if not can_perform_action(action_type):
        logger.warning(f"Daily limit reached for {action_type}: {get_daily_count(action_type)}")
        return None
    
    if not can_post_tweet():
        total_tweets = (get_daily_count('main_posts') + 
                       get_daily_count('crypto_bullposts') + 
                       get_daily_count('mentions'))
        logger.warning(f"Daily total tweet limit reached: {total_tweets}")
        return None
    
    if not can_perform_15min_action('tweets'):
        logger.warning(f"15-minute tweet limit reached: {get_15min_count('tweets')}")
        return None
        
    logger.info(f"safe_tweet: Attempting to tweet ({len(text)} chars) - Type: {action_type}")
    
    try:
        result = await safe_api_call(
            lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
            None, 0, text, media_id, **kwargs
        )
        
        if result is None:
            # Daily limit was hit during the API call
            return None
            
        increment_daily_count(action_type)
        increment_15min_count('tweets')
        logger.info(f"✅ Tweet posted successfully - {action_type} count: {get_daily_count(action_type)}, 15min tweets: {get_15min_count('tweets')}")
        return result
    except Exception as e:
        logger.error(f"safe_tweet: Error posting tweet: {e}", exc_info=True)
        raise e

async def safe_like(tweet_id: str):
    if not can_perform_action('likes'):
        logger.info(f"Daily like limit reached: {get_daily_count('likes')}")
        return None
        
    try:
        result = await safe_api_call(
            lambda tid: x_client.like(tid),  # v2 endpoint
            None, 0, tweet_id
        )
        increment_daily_count('likes')
        logger.info(f"Liked tweet {tweet_id} - Like count: {get_daily_count('likes')}")
        return result
    except Exception as e:
        logger.error(f"Error liking tweet: {e}")
        return None

async def safe_retweet(tweet_id: str):
    if not can_perform_action('retweets'):
        logger.info(f"Daily retweet limit reached: {get_daily_count('retweets')}")
        return None
    
    if not can_perform_15min_action('retweets'):
        logger.info(f"15-minute retweet limit reached: {get_15min_count('retweets')}")
        return None
        
    try:
        result = await safe_api_call(
            lambda tid: x_client.retweet(tid),  # v2 endpoint
            None, 0, tweet_id
        )
        increment_daily_count('retweets')
        increment_15min_count('retweets')
        logger.info(f"Retweeted {tweet_id} - Daily: {get_daily_count('retweets')}, 15min: {get_15min_count('retweets')}")
        return result
    except Exception as e:
        logger.error(f"Error retweeting: {e}")
        return None

async def safe_follow(user_id: str):
    if not can_perform_action('follows'):
        logger.info(f"Daily follow limit reached: {get_daily_count('follows')}")
        return None
    
    if not can_perform_15min_action('follows'):
        logger.info(f"15-minute follow limit reached: {get_15min_count('follows')}")
        return None
        
    try:
        result = await safe_api_call(
            lambda uid: x_client.follow_user(uid),  # v2 endpoint
            None, 0, user_id
        )
        increment_daily_count('follows')
        increment_15min_count('follows')
        logger.info(f"Followed user {user_id} - Daily: {get_daily_count('follows')}, 15min: {get_15min_count('follows')}")
        return result
    except Exception as e:
        logger.error(f"Error following user: {e}")
        return None

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

async def post_crypto_bullpost(tweet, is_mention=False):
    """Post a crypto bullpost reply with meme"""
    try:
        # Check 15-minute tweet limit before proceeding
        if not can_perform_15min_action('tweets'):
            logger.warning(f"15-minute tweet limit reached, skipping bullpost for tweet {tweet.id}")
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
            return
            
        convo_id = tweet.conversation_id or tweet.id
        history = get_thread_history(convo_id) if is_mention else ""
        
        # Different prompts for mentions vs crypto posts
        if is_mention:
            prompt = (
                f"History:{history}\n"
                f"User: '{tweet.text}'\n"
                "Write a one-liner bullpost for $DEGEN based on the above. "
                f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA. No slang. High class but a little edgy like Don Draper."
            )
        else:
            prompt = (
                f"User posted about crypto: '{tweet.text[:100]}...'\n"
                "Write a compelling one-liner bullpost about $DEGEN that fits this crypto conversation. "
                f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA. No slang. High class but edgy like Don Draper."
            )
        
        msg = ask_grok(prompt)
        img = choice(glob.glob("raid_images/*.jpg"))
        media_id = x_api.media_upload(img).media_id_string  # v1.1 media upload still works
        
        action_type = 'mentions' if is_mention else 'crypto_bullposts'
        
        await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            media_id=media_id,
            in_reply_to_tweet_id=tweet.id,
            action_type=action_type
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
        
    except Exception as e:
        logger.error(f"Error in post_crypto_bullpost for tweet {tweet.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    """Handle @mentions to the bot"""
    try:
        # Check 15-minute tweet limit at the start
        if not can_perform_15min_action('tweets'):
            logger.warning(f"15-minute tweet limit reached, skipping mention {tw.id}")
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
            
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
            await post_crypto_bullpost(tw, is_mention=True)
            return

        # 2) CA command (contract address only)
        if re.search(r"\bca\b", txt, re.IGNORECASE) and not re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 3) DEX command
        if re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=build_dex_reply(DEGEN_ADDR),
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 4) Token lookup
        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
        if token:
            sym = token.lstrip('$').upper()
            addr = DEGEN_ADDR if sym=="DEGEN" else lookup_address(token)
            if addr:
                await safe_tweet(
                    text=build_dex_reply(addr),
                    in_reply_to_tweet_id=tw.id,
                    action_type='mentions'
                )
                redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                return

        # 5) General response
        prompt = (
            f"History:{history}\n"
            f"User asked: \"{txt}\"\n"
            "Answer naturally and concisely."
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
        
        img = choice(glob.glob("raid_images/*.jpg"))
        media_id = x_api.media_upload(img).media_id_string
        
        await safe_tweet(
            text=reply,
            media_id=media_id,
            in_reply_to_tweet_id=tw.id,
            action_type='mentions'
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        update_thread(convo_id, txt, reply)
        increment_thread(convo_id)
        
    except Exception as e:
        logger.error(f"Error handling mention {tw.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

async def search_mentions_loop():
    """Search for @mentions"""
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
    
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
            
            search_results = await safe_search(x_client.search_recent_tweets, 'mentions', **search_params)
            if search_results:
                logger.info("Successfully searched for mentions")
            
            if search_results and search_results.data:
                for tw in search_results.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS:
                        logger.info(f"Skipping blocked tweet ID: {tw.id}")
                        continue
                        
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    logger.info(f"Processing mention: {tw.id}")
                    await handle_mention(tw)
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        await asyncio.sleep(150)  # Every 2.5 minutes

async def contract_engagement_loop():
    """Find tweets mentioning the contract address and engage with them"""
    key = f"{REDIS_PREFIX}last_contract_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    
    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": f"{DEGEN_ADDR} -is:retweet -is:reply",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "author_id", "public_metrics", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username", "public_metrics"],
                "max_results": 10
            }
            
            logger.info(f"Searching for contract address mentions: {DEGEN_ADDR}")
            res = await safe_search(x_client.search_recent_tweets, 'general', **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                users_dict = {user.id: user for user in (res.includes.get('users', []))} if res.includes else {}
                
                for tw in res.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS:
                        continue
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    if tw.author_id == BOT_ID:  # Skip our own tweets
                        continue
                    
                    user = users_dict.get(tw.author_id)
                    
                    try:
                        # Always like contract mentions
                        await safe_like(str(tw.id))
                        
                        # Retweet most of them (80% chance since they already contain contract address)
                        if randint(1, 10) <= 8:
                            await safe_retweet(str(tw.id))
                        
                        # Sometimes reply with bullpost (10% chance to avoid spam)
                        if randint(1, 10) == 1 and can_perform_action('crypto_bullposts') and can_perform_15min_action('tweets'):
                            logger.info(f"Posting bullpost on contract mention {tw.id}")
                            await post_crypto_bullpost(tw, is_mention=False)
                        
                        # Consider following active accounts with good engagement
                        if (user and user.public_metrics.get('followers_count', 0) < 10000 and 
                            randint(1, 50) == 1):  # 2% chance to follow
                            await safe_follow(str(user.id))
                        
                        logger.info(f"Engaged with contract mention: {tw.id}")
                        
                        # Mark as processed
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        
                        # Small delay between engagements
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error engaging with contract mention {tw.id}: {e}")
                        continue
                
                redis_client.set(key, str(newest))
                
        except Exception as e:
            logger.error(f"contract_engagement_loop error: {e}")
        
        await asyncio.sleep(600)  # Every 10 minutes (1.5 calls per 15min)

async def ogdegen_monitor_loop():
    """Monitor and retweet @ogdegenonsol posts"""
    key = f"{REDIS_PREFIX}last_ogdegen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    
    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "from:ogdegenonsol -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "created_at"],
                "max_results": 10
            }
            
            res = await safe_search(x_client.search_recent_tweets, 'general', **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    # Always retweet ogdegen posts
                    await safe_retweet(str(tw.id))
                    # Also like them
                    await safe_like(str(tw.id))
                    logger.info(f"Retweeted and liked @ogdegenonsol post: {tw.id}")
                
                redis_client.set(key, str(newest))
                
        except Exception as e:
            logger.error(f"ogdegen_monitor_loop error: {e}")
        
        await asyncio.sleep(300)  # Every 5 minutes

async def main_post_loop():
    """Main account posts every 80 minutes (18 times per day) promoting Ask Degen platform features"""
    
    # Much more varied prompts - mix of professional tech founder and experienced trader perspectives
    platform_prompts = [
        # Professional Tech Founder Voice (no "As Ask Degen")
        "Write a tweet about why most crypto analysis tools fail traders. Professional insight from someone who's built better solutions.",
        
        "Write a tweet about the gap between social media hype and actual market data in crypto. Tech founder perspective.",
        
        "Write a professional tweet about what separates successful meme coin platforms from the noise. Hint at having built something different.",
        
        "Write a tweet about the psychology behind pump and dump schemes. Educational tone from someone with data access.",
        
        "Write a tweet about why reading charts isn't enough for meme coins. Professional take on multi-signal analysis.",
        
        # Experienced Trader/Degen Voice
        "Write a tweet from an experienced meme coin trader about red flags most people miss in new launches.",
        
        "Write a tweet about the difference between diamond hands and smart risk management. Experienced trader perspective.",
        
        "Write a tweet about why following the right crypto voices matters more than following the loudest ones.",
        
        "Write a tweet about timing meme coin entries and exits. Share wisdom from someone who's seen many cycles.",
        
        "Write a tweet about liquidity traps that catch new meme coin traders. Educational from experience.",
        
        # Analytical/Data-Driven Voice
        "Write a tweet about patterns you notice before meme coins pump. Analytical perspective without giving away secrets.",
        
        "Write a tweet about the importance of token age analysis in meme coin research. Data-driven insight.",
        
        "Write a tweet about social sentiment vs actual trading volume in crypto markets. Analytical observation.",
        
        "Write a tweet about why Google Trends data matters for crypto timing. Research-based perspective.",
        
        "Write a tweet about measuring community authenticity vs artificial hype. Data analyst angle.",
        
        # Educator Voice
        "Write an educational tweet about what new meme coin traders should learn first. Helpful teacher tone.",
        
        "Write a tweet explaining why most crypto 'alpha' on Twitter is actually noise. Educational debunking.",
        
        "Write a tweet about the real cost of FOMO in meme coin trading. Educational with examples.",
        
        "Write a tweet about risk management basics for volatile crypto markets. Educational finance perspective.",
        
        "Write a tweet about the learning curve in crypto trading and how to shorten it.",
        
        # Market Observer Voice  
        "Write a tweet observing how meme coin narratives spread across crypto Twitter. Sociological angle.",
        
        "Write a tweet about the lifecycle of meme coin communities. Anthropological observation.",
        
        "Write a tweet about information asymmetry in crypto markets. Economic perspective.",
        
        "Write a tweet about how crypto Twitter really influences price action. Media analysis angle.",
        
        "Write a tweet about the evolution of meme coin trading over the past year. Historical perspective.",
        
        # Problem-Solver Voice
        "Write a tweet about the biggest unsolved problems in meme coin trading. Solution-oriented thinking.",
        
        "Write a tweet about why current crypto tools don't serve retail traders well. Problem identification.",
        
        "Write a tweet about what meme coin traders actually need vs what they think they need.",
        
        "Write a tweet about bridging the gap between professional trading tools and retail accessibility.",
        
        "Write a tweet about making crypto markets more transparent for everyone.",
        
        # Contrarian Voice
        "Write a contrarian take on meme coin 'diamond hands' culture. Thoughtful disagreement.",
        
        "Write a tweet challenging common wisdom about crypto influencer alpha. Contrarian but fair.",
        
        "Write a tweet about why chasing the latest meme coin meta usually fails. Contrarian education.",
        
        "Write a tweet questioning whether more crypto data actually helps most traders. Philosophical angle.",
        
        "Write a tweet about why the loudest crypto voices aren't always the smartest. Contrarian observation.",
        
        # Storyteller Voice
        "Tell a brief story about a meme coin trade that taught you an important lesson about market psychology.",
        
        "Share an observation about crypto market behavior that most people never notice.",
        
        "Describe what separates traders who survive crypto winters from those who don't.",
        
        "Share insight about how crypto communities really form and grow organically.",
        
        "Tell about a pattern in meme coin launches that took you years to recognize."
    ]
    
    # Prompt style variations to add even more diversity
    style_variations = [
        "Keep it under 200 characters. Professional but accessible.",
        "Write conversationally. Share wisdom without lecturing.", 
        "Be direct and actionable. Give practical insight.",
        "Write thoughtfully. This should make people think.",
        "Be educational but not preachy. Share knowledge naturally.",
        "Write with quiet confidence. No need to oversell the point.",
        "Keep it real. Share honest market observations.",
        "Write like you're talking to a friend who trades crypto.",
        "Be analytical but human. Data with personality.",
        "Write with earned authority. You've seen this before."
    ]
    
    post_counter = 0
    last_style_used = 0
    logger.info("Starting main_post_loop with varied Ask Degen platform promotion (18 posts/day)...")

    while True:
        try:
            if can_perform_action('main_posts') and can_post_tweet():
                logger.info(f"Ask Degen platform post attempt #{post_counter + 1}")
                
                # Enhanced REDIS tracking with better content deduplication
                today_key = f"{REDIS_PREFIX}themes_used:{time.strftime('%Y-%m-%d')}"
                content_history_key = f"{REDIS_PREFIX}content_snippets:{time.strftime('%Y-%m-%d')}"
                phrase_history_key = f"{REDIS_PREFIX}phrase_history"
                
                # Get recent content to avoid repetition
                recent_posts = redis_client.lrange(f"{REDIS_PREFIX}recent_main_posts", 0, 14)  # More history
                recent_phrases = redis_client.lrange(phrase_history_key, 0, 49)  # Track phrases
                recent_snippets = redis_client.lrange(content_history_key, 0, 29)
                
                # Select prompt with better variation
                max_attempts = 10
                best_prompt = None
                best_style = None
                
                for attempt in range(max_attempts):
                    # Use a more random selection instead of linear
                    seed = int(time.time()) + post_counter + attempt
                    prompt_index = seed % len(platform_prompts)
                    
                    # Vary style selection
                    style_index = (last_style_used + 1 + attempt) % len(style_variations)
                    
                    test_prompt = platform_prompts[prompt_index]
                    test_style = style_variations[style_index]
                    
                    # Check if this prompt combo was used recently
                    prompt_signature = f"{prompt_index}_{style_index}"
                    recent_signatures = redis_client.lrange(f"{REDIS_PREFIX}prompt_signatures", 0, 9)
                    
                    if prompt_signature.encode() not in recent_signatures:
                        best_prompt = test_prompt
                        best_style = test_style
                        last_style_used = style_index
                        
                        # Track this signature
                        redis_client.lpush(f"{REDIS_PREFIX}prompt_signatures", prompt_signature)
                        redis_client.ltrim(f"{REDIS_PREFIX}prompt_signatures", 0, 19)
                        redis_client.expire(f"{REDIS_PREFIX}prompt_signatures", 86400)
                        break
                
                if not best_prompt:
                    # Fallback if all recent prompts were used
                    best_prompt = platform_prompts[post_counter % len(platform_prompts)]
                    best_style = style_variations[post_counter % len(style_variations)]
                
                # Build context-aware prompt
                avoid_phrases = []
                for recent in recent_phrases[-20:]:  # Last 20 phrases
                    if isinstance(recent, bytes):
                        recent = recent.decode('utf-8')
                    avoid_phrases.append(recent)
                
                enhanced_prompt = f"""{best_prompt}

{best_style}

IMPORTANT CONSTRAINTS:
- Never start with "As Ask Degen" or "Ask Degen here"
- Write naturally in the specified voice/perspective 
- Under 240 chars to leave room for $DEGEN tag
- No crypto slang (no "WAGMI", "LFG", "fren", etc.)
- Educational value for meme coin traders
- Subtly reference having built tools/platforms without being salesy

AVOID these overused phrases: {', '.join(avoid_phrases[-10:]) if avoid_phrases else 'None'}

Context: You've built a platform that analyzes social sentiment, tracks KOLs, detects scams, analyzes token launches, helps traders learn. Reference capabilities naturally."""
                
                # Get content from Grok
                raw_content = ask_grok(enhanced_prompt).strip()
                
                # Clean up common repetitive starts
                cleanup_patterns = [
                    r'^As Ask Degen,?\s*',
                    r'^Ask Degen here[.,:]?\s*',
                    r'^At Ask Degen,?\s*',
                    r'^From Ask Degen[.,:]?\s*'
                ]
                
                for pattern in cleanup_patterns:
                    raw_content = re.sub(pattern, '', raw_content, flags=re.IGNORECASE)
                raw_content = raw_content.strip()
                
                # Validate content quality
                if not raw_content or len(raw_content) < 20:
                    logger.warning("Grok returned insufficient content, skipping post")
                    continue
                
                # Extract opening phrases to track repetition
                opening_phrase = ' '.join(raw_content.split()[:4])  # First 4 words
                redis_client.lpush(phrase_history_key, opening_phrase)
                redis_client.ltrim(phrase_history_key, 0, 99)  # Keep last 100
                redis_client.expire(phrase_history_key, 172800)  # 2 days
                
                # Build final tweet
                tweet = f"{raw_content}\n\n$DEGEN"
                
                # Ensure character limit
                if len(tweet) > 270:
                    tweet = truncate_to_sentence(raw_content, 240) + "\n\n$DEGEN"
                
                # Enhanced similarity check
                if is_content_too_similar_enhanced(tweet, recent_posts, recent_snippets):
                    logger.info("Content too similar to recent posts, trying different approach")
                    post_counter += 1
                    continue
                
                # Post the tweet
                result = await safe_tweet(tweet, action_type='main_posts')
                
                if result:  # Success - update all tracking
                    # Track recent posts (keep last 15)
                    redis_client.lpush(f"{REDIS_PREFIX}recent_main_posts", tweet)
                    redis_client.ltrim(f"{REDIS_PREFIX}recent_main_posts", 0, 14)
                    redis_client.expire(f"{REDIS_PREFIX}recent_main_posts", 86400)
                    
                    # Track content snippets for better deduplication
                    content_snippet = ' '.join(raw_content.split()[:8])  # First 8 words
                    redis_client.lpush(content_history_key, content_snippet)
                    redis_client.ltrim(content_history_key, 0, 49)
                    redis_client.expire(content_history_key, 86400)
                    
                    logger.info(f"Ask Degen platform post published! Style: {best_style[:30]}...")
                    
                else:
                    logger.warning("Tweet failed to post")

                post_counter += 1
                
            else:
                if not can_perform_action('main_posts'):
                    logger.info(f"Main post limit reached: {get_daily_count('main_posts')}/18")
                if not can_post_tweet():
                    total_tweets = (get_daily_count('main_posts') + 
                                   get_daily_count('crypto_bullposts') + 
                                   get_daily_count('mentions'))
                    logger.info(f"Total tweet limit reached: {total_tweets}/100")
                
        except Exception as e:
            logger.error(f"Main post error: {e}", exc_info=True)
            
        await asyncio.sleep(4800)  # 80 minutes

# Enhanced similarity detection function
def is_content_too_similar_enhanced(new_content: str, recent_posts: list, recent_snippets: list) -> bool:
    """Enhanced similarity check with multiple detection methods"""
    if not recent_posts and not recent_snippets:
        return False
    
    new_words = set(new_content.lower().split())
    new_content_lower = new_content.lower()
    
    # Check against recent full posts
    for recent_post in recent_posts[:8]:  # Check last 8 posts
        if isinstance(recent_post, bytes):
            recent_post = recent_post.decode('utf-8')
        
        recent_words = set(recent_post.lower().split())
        
        # Word overlap check
        common_words = new_words.intersection(recent_words)
        total_unique_words = len(new_words.union(recent_words))
        
        if total_unique_words > 0:
            similarity_ratio = len(common_words) / total_unique_words
            if similarity_ratio > 0.35:  # Reduced threshold
                return True
        
        # Phrase similarity check
        if any(phrase in new_content_lower for phrase in recent_post.lower().split('.') if len(phrase) > 15):
            return True
    
    # Check against content snippets
    for snippet in recent_snippets[:20]:
        if isinstance(snippet, bytes):
            snippet = snippet.decode('utf-8')
        
        if snippet.lower() in new_content_lower:
            return True
    
    return False

async def log_daily_stats():
    """Log daily statistics every hour"""
    while True:
        try:
            stats = {
                'main_posts': get_daily_count('main_posts'),
                'crypto_bullposts': get_daily_count('crypto_bullposts'),
                'mentions': get_daily_count('mentions'),
                'likes': get_daily_count('likes'),
                'retweets': get_daily_count('retweets'),
                'follows': get_daily_count('follows')
            }
            
            total_tweets = stats['main_posts'] + stats['crypto_bullposts'] + stats['mentions']
            
            fifteen_min_stats = {
                'searches': get_15min_count('searches'),
                'mentions_search': get_15min_count('mentions_search'),
                'retweets_15min': get_15min_count('retweets'),
                'follows_15min': get_15min_count('follows'),
                'tweets_15min': get_15min_count('tweets')
            }
            
            # Add daily limit exhaustion status
            global daily_tweet_limit_exhausted, daily_limit_reset_time
            exhaustion_status = ""
            if daily_tweet_limit_exhausted and daily_limit_reset_time:
                hours_until_reset = (daily_limit_reset_time - time.time()) / 3600
                exhaustion_status = f" | 🚫 TWEETS DISABLED ({hours_until_reset:.1f}h until reset)"
            
            logger.info(f"Daily Stats: {stats} | Total Tweets: {total_tweets}/100{exhaustion_status}")
            logger.info(f"15-min Stats: {fifteen_min_stats}")
            
        except Exception as e:
            logger.error(f"Stats logging error: {e}")
            
        await asyncio.sleep(3600)  # Every hour

async def main():
    # Pre-mark blocked tweets
    for tweet_id in BLOCKED_TWEET_IDS:
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tweet_id)
        logger.info(f"Pre-marked blocked tweet ID {tweet_id} as replied")
    
    logger.info("Starting all loops with EFFICIENT CONTRACT ADDRESS SEARCH...")
    logger.info("🔧 KEY CHANGES:")
    logger.info("- Contract engagement: Search directly for contract address mentions")
    logger.info("- No more wasted searches on general crypto terms")
    logger.info("- Higher retweet rate (80%) since all results contain contract address")
    logger.info("- @ogdegenonsol posts: Always retweeted")
    logger.info("- Enhanced main post variation with 40+ different prompt styles")
    logger.info("")
    logger.info("Search intervals:")
    logger.info("- Mentions: every 2.5min (6/15min, under 10 limit)")
    logger.info("- Contract engagement: every 10min (1.5/15min)")
    logger.info("- OGdegen monitor: every 5min (3/15min)")
    logger.info("Total: ~10.5 searches per 15min (well under 60 limit)")
    
    await asyncio.gather(
        search_mentions_loop(),
        main_post_loop(),
        contract_engagement_loop(),
        ogdegen_monitor_loop(),
        log_daily_stats()
    )

if __name__ == "__main__":
    asyncio.run(main())