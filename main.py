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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
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

# Credentials
API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
GROK_KEY = os.getenv("GROK_API_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

# Redis client
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Load knowledge file
try:
    with open("degen_knowledge.txt", "r", encoding="utf-8") as f:
        DEGEN_KNOWLEDGE = f.read()
except:
    DEGEN_KNOWLEDGE = ""

# Twitter clients
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
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE = re.compile(r'^[A-Za-z0-9]{43,44}$')
GROK_URL = "https://api.x.ai/v1/chat/completions"
SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?search={}"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
mentions_timestamps = deque()
tweet_timestamps = deque()

def resolve_token(q):
    s = q.upper().lstrip('$')
    if s == 'DEGEN':
        return 'DEGEN', DEGEN_ADDR
    if ADDR_RE.match(s):
        return None, s
    try:
        resp = requests.get(SEARCH_URL.format(s), timeout=10)
        resp.raise_for_status()
        for item in resp.json():
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                addr = item.get('pairAddress') or base.get('address')
                if addr:
                    return base.get('symbol'), addr
    except:
        pass
    return None, None

def fetch_data(addr=DEGEN_ADDR):
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()[0]
        base = data.get('baseToken', {})
        out = {
            'symbol': base.get('symbol'),
            'price_usd': float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h': float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link': f"https://dexscreener.com/solana/{addr}"
        }
        return out
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(data):
    return (
        f"🚀 {data['symbol']} | ${data['price_usd']:,.6f}\n"
        f"MC ${data['market_cap']:,.0f} | Vol24 ${data['volume_usd']:,.0f}\n"
        f"1h {'🟢' if data['change_1h'] >= 0 else '🔴'}{data['change_1h']:+.2f}% | "
        f"24h {'🟢' if data['change_24h'] >= 0 else '🔴'}{data['change_24h']:+.2f}%\n{data['link']}"
    )

def ask_grok(prompt):
    try:
        body = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You're a bold, witty, aggressive crypto community voice."},
                {"role": "user", "content": prompt + DEGEN_KNOWLEDGE + "\nEnd with NFA."}
            ],
            "max_tokens": 180,
            "temperature": 0.9
        }
        headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
        r = requests.post(GROK_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return "$DEGEN. NFA."

async def safe_tweet(text, media_id=None, **kwargs):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        wait = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Tweet limit hit; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        if media_id:
            return x_client.create_tweet(text=text, media_ids=[media_id], **kwargs)
        return x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get("x-rate-limit-reset", time.time() + RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_tweet(text, media_id=media_id, **kwargs)
    tweet_timestamps.append(time.time())

async def poll_loop():
    while True:
        try:
            last_id = db.get("last_mention_id")
            mentions = x_client.get_users_mentions(
                id=BOT_ID,
                since_id=last_id,
                tweet_fields=["id", "text", "author_id", "in_reply_to_user_id"],
                expansions=["author_id"],
                user_fields=["username"],
                max_results=10
            )
            if mentions and mentions.data:
                for tw in reversed(mentions.data):
                    if str(tw.in_reply_to_user_id) != str(BOT_ID):
                        continue
                    txt = tw.text.strip()
                    tid = tw.id
                    txt_lower = txt.lower()
                    
                    if "raid" in txt_lower:
                        text = ask_grok("Start a bold tweet about buying $DEGEN and tag @ogdegenonsol")
                        images = glob.glob("raid_images/*.jpg")
                        if images:
                            img = choice(images)
                            media = x_api.media_upload(img)
                            await safe_tweet(text=text, media_id=media.media_id_string)
                        else:
                            await safe_tweet(text=text)
                    elif txt_lower == "dex":
                        reply = format_metrics(fetch_data(DEGEN_ADDR))
                        await safe_tweet(text=reply.strip(), in_reply_to_tweet_id=tid)
                    elif txt_lower == "ca":
                        reply = f"Contract Address: {DEGEN_ADDR}"
                        await safe_tweet(text=reply.strip(), in_reply_to_tweet_id=tid)
                    else:
                        token = next((w for w in txt.split() if w.startswith("$") or ADDR_RE.match(w)), None)
                        if token:
                            sym, addr = resolve_token(token)
                            if addr:
                                d = fetch_data(addr)
                                reply = format_metrics(d)
                            else:
                                reply = "Token not found."
                            await safe_tweet(text=reply.strip(), in_reply_to_tweet_id=tid)
                        else:
                            reply = ask_grok(txt)
                            await safe_tweet(text=reply.strip(), in_reply_to_tweet_id=tid)
                    db.set("last_mention_id", tid)
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            d = fetch_data()
            card = format_metrics(d)
            context = ask_grok("Write a 1-sentence bullish summary of these metrics:")
            tweet = f"{card}\n{context}"
            if len(tweet) > 380:
                tweet = tweet[:380].rsplit('.', 1)[0] + '.'
            await safe_tweet(text=tweet)
            logger.info("Hourly post success")
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        poll_loop(),
        hourly_post_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
