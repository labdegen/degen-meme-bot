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

# Updated daily limits for Basic tier
DAILY_TWEET_LIMITS = {
    'main_posts': 6,        # Every 4 hours
    'crypto_bullposts': 20, # Reduced from 24 
    'mentions': 50,         # Reduced from 70
    'likes': 200,          # Keep at 200
    'retweets': 80,        # Reduced from 100
    'follows': 40,         # Reduced from 50
    'total_tweets': 100    # New: total daily tweet limit
}

# 15-minute rate limits for actions
FIFTEEN_MIN_LIMITS = {
    'retweets': 5,         # Conservative: 5 per 15min
    'follows': 3,          # Conservative: 3 per 15min
    'searches': 9          # Conservative: 9 per 15min (under 10)
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
        logger.warning(f"Twitter rate limit hit")
        
        # Try to parse rate limit reset from headers
        reset_time = None
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
            reset_header = e.response.headers.get('x-rate-limit-reset')
            if reset_header:
                try:
                    reset_time = int(reset_header)
                    sleep_duration = max(reset_time - int(time.time()) + 10, 60)  # Add 10s buffer
                    logger.warning(f"Rate limit resets at {reset_time}, sleeping for {sleep_duration}s")
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

async def safe_search(fn, *args, **kwargs):
    if not can_perform_15min_action('searches'):
        logger.warning(f"15-minute search limit reached: {get_15min_count('searches')}")
        return None
    
    result = await safe_api_call(fn, None, 0, *args, **kwargs)
    if result:
        increment_15min_count('searches')
        logger.info(f"Search completed - 15min count: {get_15min_count('searches')}")
    return result

async def safe_tweet(text: str, media_id=None, action_type='mentions', **kwargs):
    if not can_perform_action(action_type):
        logger.warning(f"Daily limit reached for {action_type}: {get_daily_count(action_type)}")
        return None
    
    if not can_post_tweet():
        total_tweets = (get_daily_count('main_posts') + 
                       get_daily_count('crypto_bullposts') + 
                       get_daily_count('mentions'))
        logger.warning(f"Daily total tweet limit reached: {total_tweets}")
        return None
        
    logger.info(f"safe_tweet: Attempting to tweet ({len(text)} chars) - Type: {action_type}")
    
    try:
        result = await safe_api_call(
            lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
            None, 0, text, media_id, **kwargs
        )
        increment_daily_count(action_type)
        logger.info(f"Tweet posted successfully - {action_type} count: {get_daily_count(action_type)}")
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
            
            search_results = await safe_search(x_client.search_recent_tweets, **search_params)
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
        
        await asyncio.sleep(240)  # Every 4 minutes (3.75 calls per 15min)

async def crypto_engagement_loop():
    """Find and engage with crypto posts"""
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
            
            res = await safe_search(x_client.search_recent_tweets, **params)
            
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
                        # Different engagement strategies
                        engagement_choice = randint(1, 10)
                        
                        if engagement_choice <= 2:  # 20% chance - Bullpost reply (reduced from 30%)
                            if can_perform_action('crypto_bullposts'):
                                logger.info(f"Posting crypto bullpost on tweet {tw.id} from @{user.username}")
                                await post_crypto_bullpost(tw, is_mention=False)
                        
                        elif engagement_choice <= 5:  # 30% chance - Like + potential retweet
                            await safe_like(str(tw.id))
                            if randint(1, 4) == 1:  # 25% of likes also get retweeted (reduced from 33%)
                                await safe_retweet(str(tw.id))
                        
                        elif engagement_choice <= 7:  # 20% chance - Just like
                            await safe_like(str(tw.id))
                        
                        # 30% chance - no action (just observe)
                        
                        # Consider following active accounts with good engagement
                        if (user.public_metrics.get('followers_count', 0) < 10000 and 
                            randint(1, 25) == 1):  # 4% chance to follow (reduced from 5%)
                            await safe_follow(str(user.id))
                        
                        # Mark as processed
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        
                        # Small delay between actions
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error engaging with tweet {tw.id}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"crypto_engagement_loop error: {e}", exc_info=True)
        
        await asyncio.sleep(1200)  # Every 20 minutes (0.75 calls per 15min)

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
            
            res = await safe_search(x_client.search_recent_tweets, **params)
            
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
        
        await asyncio.sleep(450)  # Every 7.5 minutes (2 calls per 15min)

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
            
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    if tw.author_id == BOT_ID:  # Skip our own tweets
                        continue
                    
                    # Like all contract mentions
                    await safe_like(str(tw.id))
                    
                    # Retweet some of them (40% chance, reduced from 50%)
                    if randint(1, 5) <= 2:
                        await safe_retweet(str(tw.id))
                    
                    logger.info(f"Engaged with contract mention: {tw.id}")
                
                redis_client.set(key, str(newest))
                
        except Exception as e:
            logger.error(f"contract_monitor_loop error: {e}")
        
        await asyncio.sleep(750)  # Every 12.5 minutes (1.2 calls per 15min)

async def main_post_loop():
    """Main account posts every 4 hours"""
    grok_prompts = [
        "Write a positive one-sentence analytical update on $DEGEN using recent market data. Do not mention the contract address. No slang. High class but a little edgy like Don Draper.",
        "Write a positive one-sentence hot take on $DEGEN's price action. Be edgy and risky. Do not mention the contract address. No slang. High class but a little edgy like Don Draper.",
        "Write a one sentence, cryptic message about $DEGEN that implies insider knowledge. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace.",
        "Write a one sentence, savage comment about people who haven't bought $DEGEN yet. Do not mention the contract address. No slang. High class but a little edgy like Elon Musk.",
        "Write a one sentence comparing $DEGEN to the broader crypto market. Do not mention the contract address. No slang. High class but a little edgy like Hemingway.",
        "Write a one sentence post about diamond hands and $DEGEN's future potential. Do not mention the contract address. No slang. High class but a little edgy like Hunter Thompson."
    ]
    
    hour_counter = 0
    logger.info("Starting main_post_loop...")

    while True:
        try:
            if can_perform_action('main_posts') and can_post_tweet():
                logger.info(f"Main post attempt #{hour_counter + 1}")
                
                # Fetch market data
                data = fetch_data(DEGEN_ADDR)
                if data:
                    metrics = format_metrics(data)
                    dex_link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")

                    # Ask Grok for content
                    selected_prompt = grok_prompts[hour_counter % len(grok_prompts)]
                    raw = ask_grok(selected_prompt).strip()

                    # Build tweet
                    tweet = (
                        metrics.rstrip() +
                        "\n\n" +
                        raw +
                        "\n\n" +
                        dex_link
                    )

                    # Only post if different from last
                    last = redis_client.get(f"{REDIS_PREFIX}last_main_post")
                    if tweet != last:
                        await safe_tweet(tweet, action_type='main_posts')
                        redis_client.set(f"{REDIS_PREFIX}last_main_post", tweet)
                        logger.info("Main post published successfully!")

                hour_counter += 1
            else:
                if not can_perform_action('main_posts'):
                    logger.info(f"Main post limit reached: {get_daily_count('main_posts')}")
                if not can_post_tweet():
                    total_tweets = (get_daily_count('main_posts') + 
                                   get_daily_count('crypto_bullposts') + 
                                   get_daily_count('mentions'))
                    logger.info(f"Total tweet limit reached: {total_tweets}")
                
        except Exception as e:
            logger.error(f"Main post error: {e}", exc_info=True)
            
        await asyncio.sleep(14400)  # Every 4 hours

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
                'retweets_15min': get_15min_count('retweets'),
                'follows_15min': get_15min_count('follows')
            }
            
            logger.info(f"Daily Stats: {stats} | Total Tweets: {total_tweets}/100")
            logger.info(f"15-min Stats: {fifteen_min_stats}")
            
        except Exception as e:
            logger.error(f"Stats logging error: {e}")
            
        await asyncio.sleep(3600)  # Every hour

async def main():
    # Pre-mark blocked tweets
    for tweet_id in BLOCKED_TWEET_IDS:
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tweet_id)
        logger.info(f"Pre-marked blocked tweet ID {tweet_id} as replied")
    
    logger.info("Starting all loops with rate limiting...")
    logger.info("Search intervals adjusted for <10 calls per 15min:")
    logger.info("- Mentions: every 4min (3.75/15min)")
    logger.info("- Crypto engagement: every 20min (0.75/15min)")
    logger.info("- OGdegen monitor: every 7.5min (2/15min)")
    logger.info("- Contract monitor: every 12.5min (1.2/15min)")
    logger.info("Total: ~7.7 searches per 15min")
    
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