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
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.getenv('DOTENV_PATH', '.env')
load_dotenv(dotenv_path)

# List of problematic tweet IDs to always skip
BLOCKED_TWEET_IDS = [
    "1924845778821845267", 
    "1926657606195593300", 
    "1926648154012741852"
]

# Required environment vars
required = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "OPENAI_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# OpenAI API key for fallback
env_openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = env_openai_key

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
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET"),
    wait_on_rate_limit=True  # ADDED: Handle rate limits automatically
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

# Regex patterns
ADDR_RE = re.compile(r"\b[A-Za-z0-9]{43,44}\b")
USERNAME_RE = re.compile(rf"@{BOT_USERNAME}\b", re.IGNORECASE)

# Initial search ID for timelines
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

# Daily tweet tracking limits
DAILY_TWEET_LIMITS = {
    'main_posts': 6,
    'crypto_bullposts': 24,
    'mentions': 70,
    'likes': 200,
    'retweets': 100,
    'follows': 50
}

# Helper functions
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
    key = f"{REDIS_PREFIX}daily:{action_type}:{time.strftime('%Y-%m-%d')}"
    count = redis_client.get(key)
    return int(count) if count else 0

def increment_daily_count(action_type):
    key = f"{REDIS_PREFIX}daily:{action_type}:{time.strftime('%Y-%m-%d')}"
    redis_client.incr(key)
    redis_client.expire(key, 86400)
    return get_daily_count(action_type)

def can_perform_action(action_type):
    return get_daily_count(action_type) < DAILY_TWEET_LIMITS.get(action_type, 0)

# Prompt system
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. No slang. High class but a little edgy like Don Draper. "
    f"IF DEGEN, or $DEGEN is mentioned, always ONLY speak about the $DEGEN token at contract address {DEGEN_ADDR}. DO not include the contract address in the reply. "
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
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(GROK_URL, json=payload, headers=headers, timeout=30)  # FIXED: Added timeout
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}, falling back to OpenAI…")
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=180,
                temperature=0.8,
                timeout=30  # FIXED: Added timeout
            )
            return resp.choices[0].message.content.strip()
        except Exception as oe:
            logger.error(f"OpenAI fallback also failed: {oe}")
            return "Unable to provide an update at this time."

# FIXED: Improved error handling with max retries and better timeout handling
async def safe_api_call(fn, max_retries=3, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            # Twitter API functions are already async, so call them directly with timeout
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=30.0)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"API call timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                logger.error("Max timeout retries reached")
                return None
        except (requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
        except tweepy.TooManyRequests as e:
            logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(60)  # Wait 1 minute for rate limit
        except tweepy.Forbidden as e:
            logger.error(f"Forbidden error (possibly suspended or restricted): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(2 * (attempt + 1))
    
    return None

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, 3, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, action_type='mentions', **kwargs):
    if not can_perform_action(action_type):
        logger.warning(f"Daily limit reached for {action_type}")
        return None
    
    result = await safe_api_call(
        lambda: x_client.create_tweet(text=text, media_ids=[media_id] if media_id else None, **kwargs),
        3
    )
    if result:
        increment_daily_count(action_type)
        logger.info(f"Successfully posted tweet for {action_type}")
    else:
        logger.error(f"Failed to post tweet for {action_type}")
    return result

async def safe_like(tweet_id: str):
    if not can_perform_action('likes'):
        return None
    result = await safe_api_call(
        lambda: x_client.like(tweet_id), 3
    )
    if result:
        increment_daily_count('likes')
        logger.info(f"Successfully liked tweet {tweet_id}")
    return result

async def safe_retweet(tweet_id: str):
    if not can_perform_action('retweets'):
        return None
    result = await safe_api_call(
        lambda: x_client.retweet(tweet_id), 3
    )
    if result:
        increment_daily_count('retweets')
        logger.info(f"Successfully retweeted {tweet_id}")
    return result

async def safe_follow(user_id: str):
    if not can_perform_action('follows'):
        return None
    result = await safe_api_call(
        lambda: x_client.follow_user(user_id), 3
    )
    if result:
        increment_daily_count('follows')
        logger.info(f"Successfully followed user {user_id}")
    return result

# DEX helpers
def fetch_data(addr: str) -> dict:
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=15)  # FIXED: Increased timeout
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
        r = requests.get(DEXS_SEARCH_URL + t, timeout=15)  # FIXED: Added timeout
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
    if not data:
        return f"Unable to fetch data for {addr}"
    return format_metrics(data) + data['link']

# Posting logic
def post_crypto_bullpost_template(tweet, is_mention=False):
    convo_id = tweet.conversation_id or tweet.id
    history = get_thread_history(convo_id) if is_mention else ""
    if is_mention:
        prompt = (
            f"History:{history}\n"
            f"User: '{tweet.text}'\n"
            "Write a one-liner bullpost for $DEGEN... End with NFA."
        )
    else:
        prompt = (
            f"User posted about crypto: '{tweet.text[:100]}...'\n"
            "Write a compelling one-liner bullpost about $DEGEN... End with NFA."
        )
    return prompt

async def post_crypto_bullpost(tweet, is_mention=False):
    try:
        prompt = post_crypto_bullpost_template(tweet, is_mention)
        msg = ask_grok(prompt)
        
        # Check if we have images
        image_files = glob.glob("raid_images/*.jpg")
        media_id = None
        if image_files:
            img = choice(image_files)
            try:
                media_id = x_api.media_upload(img).media_id_string
            except Exception as e:
                logger.warning(f"Failed to upload image: {e}")
        
        action_type = 'mentions' if is_mention else 'crypto_bullposts'
        result = await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            media_id=media_id,
            in_reply_to_tweet_id=tweet.id,
            action_type=action_type
        )
        
        if result:
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
            logger.info(f"Successfully posted bullpost reply to {tweet.id}")
        else:
            logger.error(f"Failed to post bullpost reply to {tweet.id}")
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))  # Mark as processed to avoid retry
            
    except Exception as e:
        logger.error(f"Error in post_crypto_bullpost: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    try:
        logger.info(f"Handling mention from tweet {tw.id}")
        convo_id = tw.conversation_id or tw.id
        if redis_client.hget(get_thread_key(convo_id), "count") is None:
            try:
                root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
                update_thread(convo_id, f"ROOT: {root}", "")
            except:
                update_thread(convo_id, f"ROOT: Unknown", "")
        
        history = get_thread_history(convo_id)
        txt = re.sub(USERNAME_RE, "", tw.text).strip()
        
        # Commands: raid, ca, dex, token lookup, else response
        if re.search(r"\braid\b", txt, re.IGNORECASE):
            await post_crypto_bullpost(tw, is_mention=True)
            return
            
        if re.search(r"\bca\b", txt, re.IGNORECASE) and not re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            result = await safe_tweet(
                text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            if result:
                redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
            
        if re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            result = await safe_tweet(
                text=build_dex_reply(DEGEN_ADDR),
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            if result:
                redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
            
        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
        if token:
            addr = DEGEN_ADDR if token.lstrip('$').upper() == 'DEGEN' else lookup_address(token)
            if addr:
                result = await safe_tweet(
                    text=build_dex_reply(addr),
                    in_reply_to_tweet_id=tw.id,
                    action_type='mentions'
                )
                if result:
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                return
                
        prompt = f"History:{history}\nUser asked: \"{txt}\"\nAnswer naturally."
        raw = ask_grok(prompt).strip()
        
        if "$DEGEN" not in raw:
            reply = f"{raw}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
        else:
            reply = raw if DEGEN_ADDR in raw else f"{raw}\n\nStack $DEGEN. ca: {DEGEN_ADDR}"
            
        if len(reply) > 360:
            reply = truncate_to_sentence(reply, 360) + f"\n\n$DEGEN. ca: {DEGEN_ADDR}"
            
        image_files = glob.glob("raid_images/*.jpg")
        media_id = None
        if image_files:
            img = choice(image_files)
            try:
                media_id = x_api.media_upload(img).media_id_string
            except Exception as e:
                logger.warning(f"Failed to upload image: {e}")
                
        result = await safe_tweet(
            text=reply,
            media_id=media_id,
            in_reply_to_tweet_id=tw.id,
            action_type='mentions'
        )
        
        if result:
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            update_thread(convo_id, txt, reply)
            increment_thread(convo_id)
            logger.info(f"Successfully handled mention {tw.id}")
        else:
            logger.error(f"Failed to handle mention {tw.id}")
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            
    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

# FIXED: Added better error handling and logging to all loops
async def search_mentions_loop():
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
    
    logger.info("Starting search_mentions_loop...")
    while True:
        try:
            logger.info("Searching for mentions...")
            params = {
                "query": f"@{BOT_USERNAME} -is:retweet",
                "tweet_fields": ["id","text","conversation_id","created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                logger.info(f"Found {len(res.data)} mentions to process")
                for tw in res.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        logger.info(f"Skipping already processed tweet {tw.id}")
                        continue
                    await handle_mention(tw)
                    await asyncio.sleep(2)  # Rate limiting between mentions
            else:
                logger.info("No new mentions found")
                
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
            
        logger.info("Mentions search complete, sleeping for 180 seconds...")
        await asyncio.sleep(180)

async def crypto_engagement_loop():
    index = 0
    logger.info("Starting crypto_engagement_loop...")
    
    while True:
        try:
            term = CRYPTO_SEARCH_TERMS[index % len(CRYPTO_SEARCH_TERMS)]
            index += 1
            logger.info(f"Searching for crypto tweets with term: {term}")
            
            params = {
                "query": f"{term} -is:retweet -is:reply",
                "tweet_fields": ["id","text","author_id","public_metrics","created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username","public_metrics"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                logger.info(f"Found {len(res.data)} crypto tweets to process")
                users = {u.id: u for u in res.includes.get('users', [])}
                scored = []
                
                for tw in res.data:
                    # ONLY process tweets that contain the DEGEN contract address
                    if DEGEN_ADDR not in tw.text: 
                        continue
                    if str(tw.id) in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    user = users.get(tw.author_id)
                    if not user or tw.author_id == BOT_ID: 
                        continue
                        
                    follow_count = user.public_metrics.get('followers_count', 0)
                    engagement = (tw.public_metrics.get('like_count',0) + tw.public_metrics.get('retweet_count',0) + tw.public_metrics.get('reply_count',0))
                    score = engagement * (2 if 1000 <= follow_count <= 50000 else (1.5 if follow_count < 1000 else 1))
                    scored.append((tw, user, score))
                    
                scored.sort(key=lambda x: x[2], reverse=True)
                logger.info(f"Processing {len(scored)} DEGEN-related tweets")
                
                for tw, user, _ in scored[:3]:
                    choice_num = randint(1,10)
                    if choice_num <= 3:
                        if can_perform_action('crypto_bullposts'):
                            await post_crypto_bullpost(tw, False)
                    elif choice_num <= 6:
                        await safe_like(str(tw.id))
                        # ONLY retweet if tweet contains contract address (which it does since we filtered above)
                        if randint(1,3) == 1:
                            await safe_retweet(str(tw.id))
                    elif choice_num <= 8:
                        await safe_like(str(tw.id))
                        
                    if user.public_metrics.get('followers_count',0) < 10000 and randint(1,20) == 1:
                        await safe_follow(str(user.id))
                        
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                    await asyncio.sleep(2)
            else:
                logger.info("No crypto tweets found")
                
        except Exception as e:
            logger.error(f"crypto_engagement_loop error: {e}", exc_info=True)
            
        logger.info("Crypto engagement complete, sleeping for 900 seconds...")
        await asyncio.sleep(900)

async def ogdegen_monitor_loop():
    key = f"{REDIS_PREFIX}last_ogdegen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    
    logger.info("Starting ogdegen_monitor_loop...")
    while True:
        try:
            logger.info("Monitoring ogdegenonsol tweets...")
            last_id = redis_client.get(key)
            params = {
                "query": "from:ogdegenonsol -is:retweet", 
                "since_id": last_id, 
                "tweet_fields": ["id","text"], 
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                logger.info(f"Found {len(res.data)} new ogdegen tweets")
                newest = max(int(t.id) for t in res.data)
                
                for tw in res.data:
                    # Create bullpost reply WITHOUT @mention in the text
                    try:
                        prompt = f"User posted about crypto: '{tw.text[:100]}...'\nWrite a compelling one-liner bullpost about $DEGEN... End with NFA."
                        msg = ask_grok(prompt)
                        
                        image_files = glob.glob("raid_images/*.jpg")
                        media_id = None
                        if image_files:
                            img = choice(image_files)
                            try:
                                media_id = x_api.media_upload(img).media_id_string
                            except Exception as e:
                                logger.warning(f"Failed to upload image: {e}")
                        
                        # Reply to the tweet (in_reply_to creates the reply thread)
                        result = await safe_tweet(
                            text=truncate_to_sentence(msg, 240),
                            media_id=media_id,
                            in_reply_to_tweet_id=tw.id,  # This makes it a reply
                            action_type='crypto_bullposts'
                        )
                        
                        if result:
                            # Always like the tweet
                            await safe_like(str(tw.id))
                            logger.info(f"Bullpost replied & liked ogdegen post: {tw.id}")
                        
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        
                    except Exception as e:
                        logger.error(f"Error replying to ogdegen: {e}")
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                    
                redis_client.set(key, str(newest))
            else:
                logger.info("No new ogdegen tweets found")
                
        except Exception as e:
            logger.error(f"ogdegen_monitor_loop error: {e}")
            
        logger.info("OG Degen monitoring complete, sleeping for 300 seconds...")
        await asyncio.sleep(300)

async def contract_monitor_loop():
    key = f"{REDIS_PREFIX}last_contract_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    
    logger.info("Starting contract_monitor_loop...")
    while True:
        try:
            logger.info("Monitoring contract address mentions...")
            last_id = redis_client.get(key)
            params = {
                "query": f"{DEGEN_ADDR} -is:retweet", 
                "since_id": last_id, 
                "tweet_fields": ["id","text","author_id","created_at"], 
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                logger.info(f"Found {len(res.data)} contract mentions")
                newest = max(int(t.id) for t in res.data)
                
                for tw in res.data:
                    if tw.author_id == BOT_ID or DEGEN_ADDR not in tw.text: 
                        continue
                        
                    await safe_like(str(tw.id))
                    if randint(1,2) == 1:
                        await safe_retweet(str(tw.id))
                    logger.info(f"Engaged with contract mention: {tw.id}")
                    
                redis_client.set(key, str(newest))
            else:
                logger.info("No new contract mentions found")
                
        except Exception as e:
            logger.error(f"contract_monitor_loop error: {e}")
            
        logger.info("Contract monitoring complete, sleeping for 600 seconds...")
        await asyncio.sleep(600)

async def main_post_loop():
    grok_prompts = [
        "Write a positive one-sentence analytical update on $DEGEN using recent market data. Do not mention the contract address. No slang. High class, secretive but a little edgy like JD Salinger. Do not mention horses, stallions or other goofy stuff.",
        "Write a positive one-sentence hot take on $DEGEN's price action. Be edgy and risky. Do not mention the contract address. No slang. High class but a little edgy like Don Draper. Do not mention horses, stallions or other goofy stuff.",
        "Write a one sentence, cryptic message about $DEGEN that implies insider knowledge. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace. Do not mention horses, stallions or other goofy stuff.",
        "Write a one sentence, savage comment about people who haven't bought $DEGEN yet. Do not mention the contract address. No slang. High class but a little edgy like Elon Musk. Do not mention horses, stallions or other goofy stuff.",
        "Write a one sentence comparing $DEGEN to the broader crypto market. Do not mention the contract address. No slang. High class but a little edgy like Hemingway. Do not mention horses, stallions or other goofy stuff.",
        "Write a one sentence post about diamond hands and $DEGEN's future potential. Do not mention the contract address. No slang. High class but a little edgy like Hunter Thompson. Do not mention horses, stallions or other goofy stuff."
    ]
    hour_counter = 0
    logger.info("Starting main_post_loop...")
    
    while True:
        try:
            logger.info("Attempting to create main post...")
            if can_perform_action('main_posts'):
                data = fetch_data(DEGEN_ADDR)
                if data:
                    metrics = format_metrics(data)
                    link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")
                    raw = ask_grok(grok_prompts[hour_counter % len(grok_prompts)]).strip()
                    tweet = f"{metrics.rstrip()}\n\n{raw}\n\n{link}"
                    
                    last_tweet = redis_client.get(f"{REDIS_PREFIX}last_main_post")
                    if tweet != last_tweet:
                        result = await safe_tweet(tweet, action_type='main_posts')
                        if result:
                            redis_client.set(f"{REDIS_PREFIX}last_main_post", tweet)
                            logger.info("Successfully posted main update")
                        else:
                            logger.error("Failed to post main update")
                    else:
                        logger.info("Tweet content unchanged, skipping duplicate")
                else:
                    logger.warning("Failed to fetch DEGEN data for main post")
            else:
                logger.info("Main post daily limit reached")
                
            hour_counter += 1
            
        except Exception as e:
            logger.error(f"Main post error: {e}", exc_info=True)
            
        logger.info("Main post attempt complete, sleeping for 14400 seconds...")
        await asyncio.sleep(14400)

async def log_daily_stats():
    logger.info("Starting daily stats logging...")
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
            logger.info(f"Daily Stats: {stats}")
        except Exception as e:
            logger.error(f"Error logging daily stats: {e}")
            
        await asyncio.sleep(3600)

async def main():
    # Pre-mark blocked tweets as processed
    for tid in BLOCKED_TWEET_IDS:
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tid)
    
    logger.info("Starting all bot loops...")
    
    # FIXED: Added individual exception handling for each task
    tasks = [
        asyncio.create_task(search_mentions_loop()),
        asyncio.create_task(main_post_loop()),
        asyncio.create_task(crypto_engagement_loop()),
        asyncio.create_task(ogdegen_monitor_loop()),
        asyncio.create_task(contract_monitor_loop()),
        asyncio.create_task(log_daily_stats())
    ]
    
    # Run tasks with individual error handling
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)