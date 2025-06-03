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

# Crypto search terms - rotating through different audiences
CRYPTO_SEARCH_TERMS = [
    "memecoin OR meme coin",
    "solana memes", 
    "crypto degen",
    "solana gems",
    "memecoin season",
    "altcoin gems",
    "crypto twitter",
    "degen plays",
    "solana alpha",
    "memecoin moonshot",
    "crypto portfolio", 
    "solana ecosystem",
    "defi gems",
    "crypto gains",
    "solana traders"
]

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

async def safe_retweet(tweet_id: str, require_contract_address: bool = False, tweet_text: str = ""):
    """
    Safe retweet function with optional contract address requirement
    
    Args:
        tweet_id: ID of tweet to retweet
        require_contract_address: If True, only retweet if tweet contains DEGEN contract address
        tweet_text: Text content of the tweet (for contract address checking)
    """
    if not can_perform_action('retweets'):
        logger.info(f"Daily retweet limit reached: {get_daily_count('retweets')}")
        return None
    
    if not can_perform_15min_action('retweets'):
        logger.info(f"15-minute retweet limit reached: {get_15min_count('retweets')}")
        return None
    
    # Check for contract address requirement
    if require_contract_address and not contains_degen_contract(tweet_text):
        logger.info(f"Skipping retweet of {tweet_id} - does not contain DEGEN contract address")
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
        
        await asyncio.sleep(150)  # Every 2.5 minutes (6 calls per 15min, under 10 limit)

async def crypto_engagement_loop():
    """Find and engage with crypto posts - ONLY retweet if contains DEGEN contract address"""
    search_term_index = 0
    
    while True:
        try:
            # Rotate through different search terms
            search_term = CRYPTO_SEARCH_TERMS[search_term_index % len(CRYPTO_SEARCH_TERMS)]
            search_term_index += 1
            
            logger.info(f"Searching crypto posts with term: {search_term}")
            
            params = {
                "query": f"{search_term} -is:retweet -is:reply",
                "tweet_fields": ["id", "text", "author_id", "public_metrics", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username", "public_metrics"],
                "max_results": 10
            }
            
            res = await safe_search(x_client.search_recent_tweets, 'general', **params)
            
            if res and res.data:
                # Sort by engagement potential (mix of follower count and recent activity)
                posts_with_scores = []
                users_dict = {user.id: user for user in (res.includes.get('users', []))}
                
                for tw in res.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS:
                        continue
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    user = users_dict.get(tw.author_id)
                    if not user:
                        continue
                    
                    # Skip if it's our own tweet
                    if tw.author_id == BOT_ID:
                        continue
                    
                    # Engagement score: mix high and low follower accounts
                    follower_count = user.public_metrics.get('followers_count', 0)
                    tweet_engagement = (tw.public_metrics.get('like_count', 0) + 
                                      tw.public_metrics.get('retweet_count', 0) + 
                                      tw.public_metrics.get('reply_count', 0))
                    
                    # Boost score for accounts with 1K-50K followers (sweet spot for engagement)
                    score = tweet_engagement
                    if 1000 <= follower_count <= 50000:
                        score *= 2
                    elif follower_count < 1000:
                        score *= 1.5  # Also good for engagement
                    
                    posts_with_scores.append((tw, user, score))
                
                # Sort by score and take top posts
                posts_with_scores.sort(key=lambda x: x[2], reverse=True)
                
                for tw, user, score in posts_with_scores[:3]:  # Top 3 posts
                    try:
                        # Very conservative engagement due to 100 tweets/day limit
                        engagement_choice = randint(1, 20)
                        
                        if engagement_choice == 1:  # 5% chance - Bullpost reply (very reduced)
                            if can_perform_action('crypto_bullposts') and can_perform_15min_action('tweets'):
                                logger.info(f"Posting crypto bullpost on tweet {tw.id} from @{user.username}")
                                await post_crypto_bullpost(tw, is_mention=False)
                        
                        elif engagement_choice <= 8:  # 35% chance - Like + potential retweet
                            await safe_like(str(tw.id))
                            # UPDATED: Only retweet if tweet contains DEGEN contract address
                            if randint(1, 5) == 1:  # 20% of likes also get retweeted
                                await safe_retweet(str(tw.id), require_contract_address=True, tweet_text=tw.text)
                        
                        elif engagement_choice <= 12:  # 20% chance - Just like
                            await safe_like(str(tw.id))
                        
                        # 40% chance - no action (just observe)
                        
                        # Consider following active accounts with good engagement
                        if (user.public_metrics.get('followers_count', 0) < 10000 and 
                            randint(1, 50) == 1):  # 2% chance to follow (very reduced)
                            await safe_follow(str(user.id))
                        
                        # Mark as processed
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        
                        # Small delay between tweet attempts to spread them out
                        await asyncio.sleep(5)
                        
                    except Exception as e:
                        logger.error(f"Error engaging with tweet {tw.id}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"crypto_engagement_loop error: {e}", exc_info=True)
        
        await asyncio.sleep(900)  # Every 15 minutes (1 call per 15min)

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
                    # Always retweet ogdegen posts (no contract address requirement)
                    await safe_retweet(str(tw.id))
                    # Also like them
                    await safe_like(str(tw.id))
                    logger.info(f"Retweeted and liked @ogdegenonsol post: {tw.id}")
                
                redis_client.set(key, str(newest))
                
        except Exception as e:
            logger.error(f"ogdegen_monitor_loop error: {e}")
        
        await asyncio.sleep(300)  # Every 5 minutes (3 calls per 15min)

async def contract_monitor_loop():
    """Monitor tweets mentioning the contract address"""
    key = f"{REDIS_PREFIX}last_contract_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    
    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": f"{DEGEN_ADDR} -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "author_id", "created_at"],
                "max_results": 10
            }
            
            res = await safe_search(x_client.search_recent_tweets, 'general', **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    if tw.author_id == BOT_ID:  # Skip our own tweets
                        continue
                    
                    # Like all contract mentions
                    await safe_like(str(tw.id))
                    
                    # Retweet some of them (30% chance, reduced from 40%)
                    # No need to check contract address here since query already filters for it
                    if randint(1, 10) <= 3:
                        await safe_retweet(str(tw.id))
                    
                    logger.info(f"Engaged with contract mention: {tw.id}")
                
                redis_client.set(key, str(newest))
                
        except Exception as e:
            logger.error(f"contract_monitor_loop error: {e}")
        
        await asyncio.sleep(450)  # Every 7.5 minutes (2 calls per 15min)

async def main_post_loop():
    """Main account posts every 80 minutes (18 times per day) with DFW-style content"""
    
    # DFW-style prompts focusing on crypto gambling lifestyle themes
    dfw_prompts = [

"Write a short one sentence Carl Sagan-style tweet about $DEGEN and crypto, decentralization and freedom. No slang. No contract address. Do not repeat content of themes from previous posts.",

"Write a short one sentence Steve Jobs-style tweet about $DEGEN and technology. No slang. No contract address. Do not repeat content of themes from previous posts.",

	"Write a short one sentence Elon Musk-style tweet about $DEGEN and crypto. Focus on risk, strategy, psychology. No slang. No contract address. Do not repeat content of themes from previous posts.",

"Write a short one sentence Steve Jobs-style tweet about $DEGEN and decentralization. No slang. No contract address. Do not repeat content of themes from previous posts.",


        "Write a short one sentence David Foster Wallace-style tweet about $DEGEN, possible themes include ai, the matrix, freedom. Focus on risk, strategy, psychology. Be cryptic subversive, mysterious. No slang. No contract address. Do not repeat content of themes from previous posts.",

"Write a short one sentence Elon Musk-style tweet about $DEGEN and crypto. Focus on risk, strategy, psychology. Be cryptic subversive, mysterious. No slang. No contract address. Do not repeat content of themes from previous posts.",

    ]
    
    # Theme categories to track and rotate through
    theme_categories = [
        "risk_sacrament", "existential_warfare", "life_changing_mindset", "volatility_paradox", 
        "alchemy_rebellion", "manifesto_graffiti", "altar_worship", "focus_blade",
        "mempool_haunting", "chaos_dance", "nerve_teaching", "rust_casino",
        "abyss_freedom", "spiritual_resistance", "signal_hunting"
    ]
    
    post_counter = 0
    logger.info("Starting main_post_loop with DFW-style content (18 posts/day)...")

    while True:
        try:
            if can_perform_action('main_posts') and can_post_tweet():
                logger.info(f"Main DFW post attempt #{post_counter + 1}")
                
                # Get current theme category and mark as used
                current_theme = theme_categories[post_counter % len(theme_categories)]
                today_key = f"{REDIS_PREFIX}themes_used:{time.strftime('%Y-%m-%d')}"
                
                # Check if this theme was used today
                theme_used_count = redis_client.hget(today_key, current_theme)
                theme_used_count = int(theme_used_count) if theme_used_count else 0
                
                # If theme used more than once today, try next themes until we find a less-used one
                attempts = 0
                while theme_used_count >= 2 and attempts < len(theme_categories):
                    post_counter += 1
                    attempts += 1
                    current_theme = theme_categories[post_counter % len(theme_categories)]
                    theme_used_count = redis_client.hget(today_key, current_theme)
                    theme_used_count = int(theme_used_count) if theme_used_count else 0
                
                # Select prompt based on current theme
                prompt_index = post_counter % len(dfw_prompts)
                selected_prompt = dfw_prompts[prompt_index]
                
                # Add theme context to prompt
                enhanced_prompt = f"{selected_prompt} Theme focus: {current_theme.replace('_', ' ')}."
                
                # Ask Grok for content
                raw_content = ask_grok(enhanced_prompt).strip()
                
                # Build final tweet - just the content + DEGEN mentions
                tweet = f"{raw_content}\n\n$DEGEN"
                
                # Ensure we don't exceed character limits
                if len(tweet) > 270:
                    tweet = truncate_to_sentence(raw_content, 240) + "\n\n$DEGEN"
                
                # Only post if different from recent posts
                recent_posts_key = f"{REDIS_PREFIX}recent_main_posts"
                recent_posts = redis_client.lrange(recent_posts_key, 0, 4)  # Check last 5 posts
                
                if tweet not in recent_posts:
                    result = await safe_tweet(tweet, action_type='main_posts')
                    
                    if result:  # Only update tracking if tweet was successful
                        # Track recent posts (keep last 10)
                        redis_client.lpush(recent_posts_key, tweet)
                        redis_client.ltrim(recent_posts_key, 0, 9)
                        redis_client.expire(recent_posts_key, 86400)
                        
                        # Track theme usage for today
                        redis_client.hincrby(today_key, current_theme, 1)
                        redis_client.expire(today_key, 86400)
                        
                        logger.info(f"DFW-style main post published! Theme: {current_theme}")
                    
                else:
                    logger.info("Skipping duplicate content, will try different theme next time")

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
            
        await asyncio.sleep(4800) 
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
    
    logger.info("Starting all loops with OFFICIAL X API rate limits...")
    logger.info("Based on X API Basic tier documentation:")
    logger.info("- Tweets: 100 per 24 hours TOTAL")
    logger.info("- Search: 60 per 15min, Mentions: 10 per 15min")
    logger.info("- Retweets/Follows: 5 per 15min each")
    logger.info("- Likes: 200 per 24 hours")
    logger.info("")
    logger.info("🔧 ENHANCED DAILY LIMIT HANDLING:")
    logger.info("- Bot will automatically detect daily limit exhaustion")
    logger.info("- Continue likes/retweets/follows but skip ALL tweets until reset")
    logger.info("- No more wasted 15-minute sleeps on exhausted limits")
    logger.info("")
    logger.info("🎯 RETWEET FILTERING:")
    logger.info("- Crypto engagement loop: ONLY retweets tweets containing DEGEN contract address")
    logger.info("- @ogdegenonsol posts: Always retweeted (no filter)")
    logger.info("- Contract mentions: Always retweeted (already filtered by search)")
    logger.info("")
    logger.info("Search intervals:")
    logger.info("- Mentions: every 2.5min (6/15min, under 10 limit)")
    logger.info("- Crypto engagement: every 15min (1/15min)")
    logger.info("- OGdegen monitor: every 5min (3/15min)")
    logger.info("- Contract monitor: every 7.5min (2/15min)")
    logger.info("Total: ~12 searches per 15min (under 60 limit)")
    logger.info("Tweet limit: 4 per 15min (to spread 100 daily tweets)")
    
    await asyncio.gather(
        search_mentions_loop(),
        main_post_loop(),
        crypto_engagement_loop(),
        ogdegen_monitor_loop(),
        contract_monitor_loop(),
        log_daily_stats()
    )

if __name__ == "__main__":
    asyncio.run(main())