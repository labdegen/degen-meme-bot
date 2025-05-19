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

# —————————————— Setup Logging ——————————————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# —————————————— Load Environment ——————————————
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY", "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
GROK_KEY = os.getenv("GROK_API_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

# —————————————— Redis Connection ——————————————
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# —————————————— Twitter Client Setup ——————————————
x_client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_KEY_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)
me = x_client.get_me().data
BOT_ID = me.id
logger.info(f"Authenticated as: {me.username} (ID: {BOT_ID})")

oauth = tweepy.OAuth1UserHandler(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
x_api = tweepy.API(oauth)

# —————————————— Constants ——————————————
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
ADDR_RE = re.compile(r'^[A-Za-z0-9]{43,44}$')

RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
mentions_timestamps = deque()
tweet_timestamps = deque()

RECENT_REPLIES_KEY = f"{REDIS_PREFIX}recent_replies"
RECENT_REPLIES_LIMIT = 50

# —————————————— Helper Functions ——————————————
def save_recent_reply(user_text, bot_text):
    entry = json.dumps({"user": user_text, "bot": bot_text})
    db.lpush(RECENT_REPLIES_KEY, entry)
    db.ltrim(RECENT_REPLIES_KEY, 0, RECENT_REPLIES_LIMIT - 1)

def get_recent_replies(n=5):
    items = db.lrange(RECENT_REPLIES_KEY, 0, n-1)
    return [json.loads(x) for x in items]

def get_thread_key(convo_id):
    return f"{REDIS_PREFIX}thread:{convo_id}"

def get_convo_count(convo_id):
    return int(db.hget(get_thread_key(convo_id), "count") or 0)

def increment_convo_count(convo_id):
    db.hincrby(get_thread_key(convo_id), "count", 1)
    db.expire(get_thread_key(convo_id), 86400)

def get_thread_history(convo_id):
    return db.hget(get_thread_key(convo_id), "history") or ""

def update_thread_history(convo_id, user_text, bot_text):
    history = get_thread_history(convo_id)
    new_history = (history + f"\nUser: {user_text}\nBot: {bot_text}")[-1000:]
    db.hset(get_thread_key(convo_id), "history", new_history)
    db.expire(get_thread_key(convo_id), 86400)

async def safe_mention_lookup(fn, *args, **kwargs):
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        wait = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Mentions limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        res = fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] get_users_mentions 429; backing off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_mention_lookup(fn, *args, **kwargs)
    mentions_timestamps.append(time.time())
    return res

async def safe_tweet(text: str, **kwargs):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        wait = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Tweet limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] create_tweet 429; backing off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_tweet(text=text, **kwargs)
    tweet_timestamps.append(time.time())
    return resp

# —————————————— AI Query Functions ——————————————
def ask_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    body = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": (
                "You are a crypto analyst: concise, sharp, professional, and a bit edgy. "
                "Always answer ONLY about the $DEGEN token on Solana (contract " + DEGEN_ADDR + "). "
                "Never mention other chains or tokens. Provide positive updates focused on Solana DEX performance."
            )},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    try:
        r = requests.post(PERPLEXITY_URL, json=body, headers=headers, timeout=25)
        logger.debug(f"Perplexity status: {r.status_code}, body: {r.text}")
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")
        return ask_grok(prompt)

def ask_grok(prompt):
    history_key = f"{REDIS_PREFIX}grok_history"
    past = set(db.lrange(history_key, 0, -1))
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": (
                "You are a crypto analyst: concise, sharp, professional, and a bit edgy. "
                "Always answer ONLY about the $DEGEN token on Solana (contract " + DEGEN_ADDR + "). "
                "Never mention other chains or tokens. Provide positive updates focused on Solana DEX performance."
            )},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=25)
        r.raise_for_status()
        reply = r.json()['choices'][0]['message']['content'].strip()
        if reply not in past:
            db.lpush(history_key, reply)
            db.ltrim(history_key, 0, 25)
        return reply
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return "Unable to provide an update at this time."

# —————————————— Data Fetching ——————————————
def fetch_data(addr=DEGEN_ADDR):
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        base = data.get('baseToken', {})
        return {
            'symbol': base.get('symbol', 'DEGEN'),
            'price_usd': float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h1', 0)),  # last hour volume
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h': float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link': f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(d):
    return (
        f"🚀 {d.get('symbol')} | ${d.get('price_usd'):, .6f}\n"
        f"Vol1h ${d.get('volume_usd'):, .0f} | MC ${d.get('market_cap'):, .0f}\n"
        f"1h {'🟢' if d.get('change_1h') >= 0 else '🔴'}{d.get('change_1h'):+.2f}% "
        f"24h {'🟢' if d.get('change_24h') >= 0 else '🔴'}{d.get('change_24h'):+.2f}%\n"
        f"{d.get('link')}"
    )

# —————————————— Interaction Handlers ——————————————
async def post_raid(tweet):
    # ... unchanged raid logic ...
    pass

async def handle_mention(tw):
    # ... unchanged mention logic ...
    pass

# —————————————— Main Loops ——————————————
async def mention_loop():
    # ... unchanged mention loop ...
    pass

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            # Craft a bulletproof prompt focused on Solana DEX data for the correct contract
            prompt = (
                f"Using only the following Solana DEX metrics for $DEGEN (contract {DEGEN_ADDR}) over the last hour: "
                f"{json.dumps(data)}, write a positive, punchy one-sentence update that highlights the token's performance on the DEX in the past hour. "
                "Do not mention any other chain or token."
            )
            tweet = ask_perplexity(prompt)
            final = f"{metrics}\n\n{tweet}"

            last = db.get(f"{REDIS_PREFIX}last_hourly_post")
            if final.strip() != last:
                await safe_tweet(text=final[:560])
                db.set(f"{REDIS_PREFIX}last_hourly_post", final.strip())
                logger.info("Hourly post success")
            else:
                logger.info("Skipped duplicate hourly post.")
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        mention_loop(),
        hourly_post_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
