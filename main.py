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
logging.basicConfig(level=logging.INFO)
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
        r = requests.post(GROK_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}, falling back to OpenAI…")
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=180,
                temperature=0.8
            )
            return resp.choices[0].message.content.strip()
        except Exception as oe:
            logger.error(f"OpenAI fallback also failed: {oe}")
            return "Unable to provide an update at this time."

async def safe_api_call(fn, timestamps_queue, limit, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
        logger.warning(f"Network error: {e}. Retrying in 5s…")
        await asyncio.sleep(5)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.TooManyRequests:
        await asyncio.sleep(60)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, None, 0, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, action_type='mentions', **kwargs):
    if not can_perform_action(action_type):
        logger.warning(f"Daily limit reached for {action_type}")
        return
    result = await safe_api_call(
        lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
        None, 0, text, media_id, **kwargs
    )
    increment_daily_count(action_type)
    return result

async def safe_like(tweet_id: str):
    if not can_perform_action('likes'):
        return
    result = await safe_api_call(
        lambda tid: x_client.like(tid), None, 0, tweet_id
    )
    increment_daily_count('likes')
    return result

async def safe_retweet(tweet_id: str):
    if not can_perform_action('retweets'):
        return
    result = await safe_api_call(
        lambda tid: x_client.retweet(tid), None, 0, tweet_id
    )
    increment_daily_count('retweets')
    return result

async def safe_follow(user_id: str):
    if not can_perform_action('follows'):
        return
    result = await safe_api_call(
        lambda uid: x_client.follow_user(uid), None, 0, user_id
    )
    increment_daily_count('follows')
    return result

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
        img = choice(glob.glob("raid_images/*.jpg"))
        media_id = x_api.media_upload(img).media_id_string
        action_type = 'mentions' if is_mention else 'crypto_bullposts'
        await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            media_id=media_id,
            in_reply_to_tweet_id=tweet.id,
            action_type=action_type
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
    except Exception as e:
        logger.error(f"Error in post_crypto_bullpost: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    try:
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
            await safe_tweet(
                text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
        if re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=build_dex_reply(DEGEN_ADDR),
                in_reply_to_tweet_id=tw.id,
                action_type='mentions'
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return
        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
        if token:
            addr = DEGEN_ADDR if token.lstrip('$').upper() == 'DEGEN' else lookup_address(token)
            if addr:
                await safe_tweet(
                    text=build_dex_reply(addr),
                    in_reply_to_tweet_id=tw.id,
                    action_type='mentions'
                )
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
        logger.error(f"Error handling mention: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

async def search_mentions_loop():
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
    while True:
        try:
            params = {
                "query": f"@{BOT_USERNAME} -is:retweet",
                "tweet_fields": ["id","text","conversation_id","created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                for tw in res.data:
                    if str(tw.id) in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        await asyncio.sleep(180)

async def crypto_engagement_loop():
    index = 0
    while True:
        try:
            term = CRYPTO_SEARCH_TERMS[index % len(CRYPTO_SEARCH_TERMS)]
            index += 1
            params = {
                "query": f"{term} -is:retweet -is:reply",
                "tweet_fields": ["id","text","author_id","public_metrics","created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username","public_metrics"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                users = {u.id: u for u in res.includes.get('users', [])}
                scored = []
                for tw in res.data:
                    if DEGEN_ADDR not in tw.text: continue
                    if str(tw.id) in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    user = users.get(tw.author_id)
                    if not user or tw.author_id == BOT_ID: continue
                    follow_count = user.public_metrics.get('followers_count', 0)
                    engagement = (tw.public_metrics.get('like_count',0) + tw.public_metrics.get('retweet_count',0) + tw.public_metrics.get('reply_count',0))
                    score = engagement * (2 if 1000 <= follow_count <= 50000 else (1.5 if follow_count < 1000 else 1))
                    scored.append((tw, user, score))
                scored.sort(key=lambda x: x[2], reverse=True)
                for tw, user, _ in scored[:3]:
                    choice_num = randint(1,10)
                    if choice_num <= 3:
                        if can_perform_action('crypto_bullposts'):
                            await post_crypto_bullpost(tw, False)
                    elif choice_num <= 6:
                        await safe_like(str(tw.id))
                        if randint(1,3) == 1:
                            await safe_retweet(str(tw.id))
                    elif choice_num <= 8:
                        await safe_like(str(tw.id))
                    if user.public_metrics.get('followers_count',0) < 10000 and randint(1,20) == 1:
                        await safe_follow(str(user.id))
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                    await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"crypto_engagement_loop error: {e}", exc_info=True)
        await asyncio.sleep(900)

async def ogdegen_monitor_loop():
    key = f"{REDIS_PREFIX}last_ogdegen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    while True:
        try:
            last_id = redis_client.get(key)
            params = {"query": "from:ogdegenonsol -is:retweet", "since_id": last_id, "tweet_fields": ["id","text"], "max_results": 10}
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    if DEGEN_ADDR not in tw.text: continue
                    await safe_retweet(str(tw.id))
                    await safe_like(str(tw.id))
                    logger.info(f"Retweeted & liked ogdegen post: {tw.id}")
                redis_client.set(key, str(newest))
        except Exception as e:
            logger.error(f"ogdegen_monitor_loop error: {e}")
        await asyncio.sleep(300)

async def contract_monitor_loop():
    key = f"{REDIS_PREFIX}last_contract_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)
    while True:
        try:
            last_id = redis_client.get(key)
            params = {"query": f"{DEGEN_ADDR} -is:retweet", "since_id": last_id, "tweet_fields": ["id","text","author_id","created_at"], "max_results": 10}
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    if tw.author_id == BOT_ID or DEGEN_ADDR not in tw.text: continue
                    await safe_like(str(tw.id))
                    if randint(1,2) == 1:
                        await safe_retweet(str(tw.id))
                    logger.info(f"Engaged with contract mention: {tw.id}")
                redis_client.set(key, str(newest))
        except Exception as e:
            logger.error(f"contract_monitor_loop error: {e}")
        await asyncio.sleep(600)

async def main_post_loop():
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
            if can_perform_action('main_posts'):
                data = fetch_data(DEGEN_ADDR)
                if data:
                    metrics = format_metrics(data)
                    link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")
                    raw = ask_grok(grok_prompts[hour_counter % len(grok_prompts)]).strip()
                    tweet = f"{metrics.rstrip()}\n\n{raw}\n\n{link}"
                    last_tweet = redis_client.get(f"{REDIS_PREFIX}last_main_post")
                    if tweet != last_tweet:
                        await safe_tweet(tweet, action_type='main_posts')
                        redis_client.set(f"{REDIS_PREFIX}last_main_post", tweet)
                hour_counter += 1
        except Exception as e:
            logger.error(f"Main post error: {e}", exc_info=True)
        await asyncio.sleep(14400)

async def log_daily_stats():
    while True:
        stats = {
            'main_posts': get_daily_count('main_posts'),
            'crypto_bullposts': get_daily_count('crypto_bullposts'),
            'mentions': get_daily_count('mentions'),
            'likes': get_daily_count('likes'),
            'retweets': get_daily_count('retweets'),
            'follows': get_daily_count('follows')
        }
        logger.info(f"Daily Stats: {stats}")
        await asyncio.sleep(3600)

async def main():
    for tid in BLOCKED_TWEET_IDS:
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tid)
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
