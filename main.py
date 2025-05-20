import os
import re
import time
import glob
import logging
import asyncio
import requests

from collections import deque
from random import choice
from dotenv import load_dotenv

import tweepy
import redis

# ——— CONFIG & SETUP —————————————————————————————

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required env vars
required = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# Twitter API (v1.1 and v2)
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

# Bot identity
me = x_client.get_me().data
BOT_ID = me.id
BOT_USERNAME = me.username
logger.info(f"Authenticated as @{BOT_USERNAME} (ID: {BOT_ID})")

# Redis for state
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
redis_client.ping()
logger.info("🔑 Redis connected")

# Constants
REDIS_PREFIX    = "degen:"
DEGEN_ADDR      = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL        = "https://api.x.ai/v1/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL        = "https://api.dexscreener.com/token-pairs/v1/solana/"
SYSTEM_PROMPT   = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. "
    f"Always speak about the $DEGEN token at contract address {DEGEN_ADDR}. "
    "Do NOT mention any other token or chain."
)

ADDR_RE   = re.compile(r"\b[A-Za-z0-9]{43,44}\b")
SYMBOL_RE = re.compile(r"\$([A-Za-z0-9]{2,10})", re.IGNORECASE)

# Rate limits & queues
RATE_WINDOW    = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT   = 50
SEARCH_LIMIT   = 10

mentions_q = deque()
tweets_q   = deque()
search_q   = deque()

# since_id workaround for community search
INITIAL_SEARCH_ID = str(((int(time.time()*1000) - 1_728_000_000) << 22))

# ——— UTILITIES ———————————————————————————————————

def truncate_to_sentence(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    s = text[:max_length]
    for sep in (". ", "! ", "? "):
        idx = s.rfind(sep)
        if idx != -1:
            return s[:idx+1]
    return s


def get_thread_key(cid):     return f"{REDIS_PREFIX}thread:{cid}"
def get_thread_history(cid): return redis_client.hget(get_thread_key(cid), "history") or ""
def increment_thread(cid):
    redis_client.hincrby(get_thread_key(cid), "count", 1)
    redis_client.expire(get_thread_key(cid), 86400)
def update_thread(cid, user_text, bot_text):
    hist  = get_thread_history(cid)
    entry = f"\nUser: {user_text}\nBot: {bot_text}"
    new_h = (hist + entry)[-2000:]
    redis_client.hset(get_thread_key(cid), "history", new_h)
    redis_client.expire(get_thread_key(cid), 86400)


def ask_grok(prompt: str) -> str:
    resp = requests.post(
        GROK_URL,
        json={
            "model": "grok-3-latest",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
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


def fetch_data(addr: str) -> dict:
    resp = requests.get(f"{DEXS_URL}{addr}", timeout=10)
    resp.raise_for_status()
    j = resp.json()
    data = j[0] if isinstance(j, list) else j
    base = data.get("baseToken", {})
    return {
        "symbol":     base.get("symbol", "DEGEN"),
        "price_usd":  float(data.get("priceUsd", 0)),
        "volume_usd": float(data.get("volume", {}).get("h24", 0)),
        "market_cap": float(data.get("marketCap", 0)),
        "change_1h":  float(data.get("priceChange", {}).get("h1", 0)),
        "change_24h": float(data.get("priceChange", {}).get("h24", 0)),
        "link":       f"https://dexscreener.com/solana/{addr}"
    }


def format_metrics(d: dict) -> str:
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%\n"
    )


def lookup_address(token: str) -> str | None:
    t = token.lstrip("$")
    if t.upper() == "DEGEN":
        return DEGEN_ADDR
    if ADDR_RE.fullmatch(t):
        return t
    try:
        resp = requests.get(DEXS_SEARCH_URL + t, timeout=10)
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

# ——— ASYNC WRAPPERS —————————————————————————————

async def safe_api_call(fn, queue: deque, limit: int, *args, **kwargs):
    now = time.time()
    while queue and now - queue[0] > RATE_WINDOW:
        queue.popleft()
    if len(queue) >= limit:
        wait = RATE_WINDOW - (now - queue[0]) + 1
        logger.info(f"Rate limit for {fn.__name__}, sleeping {wait:.1f}s")
        await asyncio.sleep(wait)
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get("x-rate-limit-reset", time.time()+RATE_WINDOW))
        await asyncio.sleep(reset - time.time() + 1)
        return await safe_api_call(fn, queue, limit, *args, **kwargs)
    finally:
        queue.append(time.time())

async def safe_mention_lookup(fn, **kw):
    return await safe_api_call(fn, mentions_q, MENTIONS_LIMIT, **kw)

async def safe_search(fn, **kw):
    return await safe_api_call(fn, search_q, SEARCH_LIMIT, **kw)

async def safe_tweet(text: str, media_id=None, **kw):
    def _post(t, m, **kw2):
        return x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw2)
    return await safe_api_call(_post, tweets_q, TWEETS_LIMIT, text, media_id, **
