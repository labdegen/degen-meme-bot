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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
load_dotenv()
for v in dotenv_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

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
logger.info(f"Authenticated as: {me.username} (ID: {BOT_ID})")

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

RATE_WINDOW = 900  # seconds
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50\mentions_timestamps = deque()
 tweet_timestamps = deque()

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
def get_thread_key(cid): return f"{REDIS_PREFIX}thread:{cid}"
def get_thread_history(cid): return redis_client.hget(get_thread_key(cid), "history") or ""
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
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. "
    f"Always speak about the $DEGEN token at contract address {DEGEN_ADDR}. "
    "Do NOT mention any other token or chain."
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
        logger.warning(f"Grok error: {e}")
        return "Unable to provide an update at this time."

async def safe_mention_lookup(fn, *args, **kwargs):
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        await asyncio.sleep(RATE_WINDOW - (now - mentions_timestamps[0]) + 1)
    try:
        return fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_mention_lookup(fn, *args, **kwargs)
    finally:
        mentions_timestamps.append(time.time())

async def safe_tweet(text: str, media_id=None, **kwargs):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        await asyncio.sleep(RATE_WINDOW - (now - tweet_timestamps[0]) + 1)
    try:
        if media_id:
            return x_client.create_tweet(text=text, media_ids=[media_id], **kwargs)
        return x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_tweet(text=text, media_id=media_id, **kwargs)
    finally:
        tweet_timestamps.append(time.time())

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

# Address lookup for non-DEGEN tokens
def lookup_address(token: str) -> str:
    t = token.lstrip('$')
    if t.upper() == 'DEGEN':
        return DEGEN_ADDR
    if ADDR_RE.fullmatch(t):
        return t
    try:
        r = requests.get(DEXS_SEARCH_URL + t, timeout=10)
        r.raise_for_status()
        results = r.json().get('tokens', [])
        for item in results:
            if item.get('symbol', '').lower() == t.lower():
                return item.get('contractAddress')
        if results:
            return results[0].get('contractAddress')
    except Exception:
        pass
    return None

# Build DEX reply with preview link
def build_dex_reply(addr: str) -> str:
    data = fetch_data(addr)
    return format_metrics(data) + data['link']

# Raid feature
async def post_raid(tweet):
    prompt = (
        f"Write a one-liner bullpost for $DEGEN based on:\n'{tweet.text}'\n"
        f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA."
    )
    msg = ask_grok(prompt)
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    await safe_tweet(truncate_to_sentence(msg,240), media_id=media_id, in_reply_to_tweet_id=tweet.id)
    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

# Handle mentions
async def handle_mention(tw):
    convo_id = tw.conversation_id or tw.id
    if redis_client.hget(get_thread_key(convo_id), "count") is None:
        root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
        update_thread(convo_id, f"ROOT: {root}", "")
    history = get_thread_history(convo_id)
    txt = tw.text.replace("@askdegen", "").strip()

    # 1) raid
    if re.search(r"\braid\b", txt, re.IGNORECASE):
        await post_raid(tw)
        return

    # 2) token/address -> DEX preview
    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    if token:
        sym = token.lstrip('$').upper()
        addr = DEGEN_ADDR if sym=="DEGEN" else lookup_address(token)
        if addr:
            await safe_tweet(build_dex_reply(addr), in_reply_to_tweet_id=tw.id)
            return

    # 3) CA / DEX commands
    if txt.upper() in ("CA","DEX"):
        await safe_tweet(build_dex_reply(DEGEN_ADDR), in_reply_to_tweet_id=tw.id)
        return

    # 4) general fallback
    prompt = (
        f"History:{history}\nUser asked: \"{txt}\"\n"
        "First, answer naturally and concisely. "
        "Then, in a second gambler-style line, segue with a fresh tagline about stacking $DEGEN. End with NFA."
    )
    raw = ask_grok(prompt)
    reply = truncate_to_sentence(raw,200) + f" Contract Address: {DEGEN_ADDR}"
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    await safe_tweet(reply, media_id=media_id, in_reply_to_tweet_id=tw.id)
    update_thread(convo_id, txt, reply)
    increment_thread(convo_id)

# Loops
async def mention_loop():
    while True:
        try:
            last = redis_client.get(f"{REDIS_PREFIX}last_mention_id")
            params = {"id":BOT_ID,"tweet_fields":["id","text","conversation_id"],
                      "expansions":["author_id"],"user_fields":["username"],"max_results":10}
            if last: params["since_id"] = int(last)
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            if res and res.data:
                for tw in reversed(res.data):
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)): continue
                    redis_client.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            raw = ask_grok("Write a one-sentence bullpost update on $DEGEN. Be promotional.")
            tweet = truncate_to_sentence(metrics+raw,560)
            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet!=last:
                img = choice(glob.glob("raid_images/*.jpg"))
                media_id = x_api.media_upload(img).media_id_string
                await safe_tweet(tweet, media_id=media_id)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(mention_loop(), hourly_post_loop())

if __name__ == "__main__":
    asyncio.run(main())
