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

API_KEY             = os.getenv("X_API_KEY")
API_KEY_SECRET      = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN        = os.getenv("X_BEARER_TOKEN")
GROK_KEY            = os.getenv("GROK_API_KEY")
PERP_KEY            = os.getenv("PERPLEXITY_API_KEY")

db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

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

# Constants
REDIS_PREFIX    = "degen:"
DEGEN_ADDR      = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
PERP_URL        = "https://api.perplexity.ai/chat/completions"
GROK_URL        = "https://api.x.ai/v1/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL        = "https://api.dexscreener.com/token-pairs/v1/solana/"

ADDR_RE         = re.compile(r'\b[A-Za-z0-9]{43,44}\b')
SYMBOL_RE       = re.compile(r'\$([A-Za-z0-9]{2,10})', re.IGNORECASE)

RATE_WINDOW     = 900
MENTIONS_LIMIT  = 10
TWEETS_LIMIT    = 50
mentions_timestamps = deque()
tweet_timestamps    = deque()

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
def get_thread_history(cid): return db.hget(get_thread_key(cid),"history") or ""
def increment_thread(cid):
    db.hincrby(get_thread_key(cid),"count",1)
    db.expire(get_thread_key(cid),86400)
def update_thread(cid,u,b):
    h=get_thread_history(cid)
    e=f"\nUser: {u}\nBot: {b}"
    new=(h+e)[-2000:]
    db.hset(get_thread_key(cid),"history",new)
    db.expire(get_thread_key(cid),86400)

# System prompt
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. "
    f"Always speak about the $DEGEN token at contract address {DEGEN_ADDR}. "
    "Do NOT mention any other token or chain."
)

def ask_primary(prompt: str) -> str:
    """Use Perplexity first, fallback to Grok."""
    body = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {PERP_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(PERP_URL, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity error: {e}; falling back to Grok")
        return ask_grok(prompt)

def ask_grok(prompt: str) -> str:
    body = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}")
        return "Unable to provide an update at this time."

async def safe_mention_lookup(fn, *a, **kw):
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        await asyncio.sleep(RATE_WINDOW - (now - mentions_timestamps[0]) + 1)
    try:
        return fn(*a, **kw)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_mention_lookup(fn, *a, **kw)
    finally:
        mentions_timestamps.append(time.time())

async def safe_tweet(text, media_id=None, **kw):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        await asyncio.sleep(RATE_WINDOW - (now - tweet_timestamps[0]) + 1)
    try:
        if media_id:
            return x_client.create_tweet(text=text, media_ids=[media_id], **kw)
        return x_client.create_tweet(text=text, **kw)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_tweet(text, media_id, **kw)
    finally:
        tweet_timestamps.append(time.time())

def fetch_data(addr: str) -> dict:
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()[0] if isinstance(r.json(), list) else r.json()
        base = data.get('baseToken', {})
        return {
            'symbol':     base.get('symbol', 'DEGEN'),
            'price_usd':  float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h':  float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link':       f"https://dexscreener.com/solana/{addr}"
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
        f"{d['link']}"
    )

def lookup_address(q: str) -> str:
    if q.lower() == 'degen':
        return DEGEN_ADDR
    if ADDR_RE.fullmatch(q):
        return q
    try:
        r = requests.get(DEXS_SEARCH_URL + q, timeout=10)
        r.raise_for_status()
        toks = r.json().get('tokens', [])
        for t in toks:
            if t.get('symbol','').lower() == q.lower():
                return t.get('contractAddress')
        if toks:
            return toks[0].get('contractAddress')
    except:
        pass
    return None

async def post_raid(tweet):
    prompt = (
        f"Write a one-liner bullpost for $DEGEN based on:\n'{tweet.text}'\n"
        f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA."
    )
    msg = ask_primary(prompt)
    img = choice(glob.glob("raid_images/*.jpg"))
    mid = x_api.media_upload(img).media_id_string
    await safe_tweet(truncate_to_sentence(msg,240), media_id=mid,
                     in_reply_to_tweet_id=tweet.id)
    db.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    convo_id = tw.conversation_id or tw.id
    # record root tweet on first mention
    if db.hget(get_thread_key(convo_id), "count") is None:
        root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
        update_thread(convo_id, f"ROOT: {root}", "")
    history = get_thread_history(convo_id)
    txt = tw.text.replace("@askdegen","").strip()

    # 1) raid
    if re.search(r"\braid\b", txt, re.IGNORECASE):
        await post_raid(tw)
        return

    # 2) token/address → metrics + preview link
    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    if token:
        sym = token.lstrip('$').upper()
        addr = DEGEN_ADDR if sym=="DEGEN" else (
            lookup_address(sym) if token.startswith('$') else token
        )
        if addr:
            data = fetch_data(addr)
            text = f"{format_metrics(data)}\n{data['link']}"
            await safe_tweet(text, in_reply_to_tweet_id=tw.id)
            return

    # 3) CA
    if txt.upper() == "CA":
        await safe_tweet(f"Contract Address: {DEGEN_ADDR}",
                         in_reply_to_tweet_id=tw.id)
        return

    # 4) DEX (explicit)
    if txt.upper() == "DEX":
        data = fetch_data(DEGEN_ADDR)
        text = f"{format_metrics(data)}\n{data['link']}"
        await safe_tweet(text, in_reply_to_tweet_id=tw.id)
        return

    # 5) general: context + unique segue + CA + meme
    prompt = (
        f"History:\n{history}\n\n"
        f"User asked: \"{txt}\"\n\n"
        "1) Answer in one concise sentence.\n"
        "2) Then, in a fresh gambler-style line, say something like "
        "\"In the meantime, I'm loading up more $DEGEN.\" "
        "3) End with NFA."
    )
    raw = ask_primary(prompt)
    body = truncate_to_sentence(raw, 200)
    reply = f"{body} Contract Address: {DEGEN_ADDR}"

    img = choice(glob.glob("raid_images/*.jpg"))
    mid = x_api.media_upload(img).media_id_string
    await safe_tweet(reply, media_id=mid, in_reply_to_tweet_id=tw.id)

    update_thread(convo_id, txt, reply)
    increment_thread(convo_id)

async def mention_loop():
    while True:
        try:
            last_id = db.get(f"{REDIS_PREFIX}last_mention_id")
            params = {
                "id": BOT_ID,
                "tweet_fields": ['id','text','conversation_id'],
                "expansions": ['author_id'],
                "user_fields": ['username'],
                "max_results": 10
            }
            if last_id:
                params["since_id"] = int(last_id)
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            if res and res.data:
                for tw in reversed(res.data):
                    if db.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    db.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            prompt = "Write a one-sentence bullposting update on $DEGEN. Be promotional."
            raw = ask_primary(prompt)
            tweet = truncate_to_sentence(f"{metrics}\n\n{raw}", 560)
            last = db.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet.strip() != last:
                img = choice(glob.glob("raid_images/*.jpg"))
                mid = x_api.media_upload(img).media_id_string
                await safe_tweet(tweet, media_id=mid)
                db.set(f"{REDIS_PREFIX}last_hourly_post", tweet.strip())
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
