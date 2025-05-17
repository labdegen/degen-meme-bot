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

# === Functions ===
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
        history_key = f"{REDIS_PREFIX}grok_history"
        past_prompts = db.lrange(history_key, 0, -1)
        body = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You're a bold, witty, aggressive crypto community voice."},
                {"role": "user", "content": prompt + DEGEN_KNOWLEDGE + "\nEnd with NFA."}
            ],
            "max_tokens": 180,
            "temperature": 0.95
        }
        headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
        r = requests.post(GROK_URL, json=body, headers=headers)
        r.raise_for_status()
        reply = r.json()['choices'][0]['message']['content'].strip()
        if reply not in past_prompts:
            db.lpush(history_key, reply)
            db.ltrim(history_key, 0, 25)
        return reply
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return "$DEGEN. NFA."

# === Event Loop ===
async def hourly_post_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            if not d: continue
            metrics = format_metrics(d)
            prompt = f"Here are the latest metrics for $DEGEN: {json.dumps(d)}"
            db.lpush(f"{REDIS_PREFIX}context_hourly", prompt)
            db.ltrim(f"{REDIS_PREFIX}context_hourly", 0, 10)
            tweet = ask_grok(prompt)
            final = f"{metrics}\n{tweet}"
            x_client.create_tweet(text=final[:380])
            logger.info("Hourly post success")
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def mention_loop():
    last_id = db.get(f"{REDIS_PREFIX}last_mention_id")
    while True:
        try:
            res = x_client.get_users_mentions(
                id=BOT_ID,
                since_id=last_id,
                tweet_fields=['id', 'text', 'author_id'],
                expansions=['author_id'],
                user_fields=['username'],
                max_results=10
            )
            if res.data:
                for tweet in reversed(res.data):
                    tid = tweet.id
                    txt = tweet.text.replace('@askdegen', '').strip()
                    db.set(f"{REDIS_PREFIX}last_mention_id", tid)
                    db.lpush(f"{REDIS_PREFIX}context_mentions", txt)
                    db.ltrim(f"{REDIS_PREFIX}context_mentions", 0, 25)

                    if 'RAID' in txt.upper():
                        grok_txt = ask_grok("Generate a bold call to raid $DEGEN. Tag @ogdegenonsol.")
                        img_list = glob.glob("memes/*.jpg")
                        media = x_api.media_upload(choice(img_list)) if img_list else None
                        x_client.create_tweet(text=grok_txt, media_ids=[media.media_id_string] if media else None)
                        continue

                    if txt.upper() == "DEX":
                        d = fetch_data(DEGEN_ADDR)
                        msg = format_metrics(d)
                    elif txt.upper() == "CA":
                        msg = f"Contract Address: {DEGEN_ADDR}"
                    else:
                        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
                        if token:
                            sym, addr = resolve_token(token)
                            if addr:
                                d = fetch_data(addr)
                                msg = format_metrics(d)
                            else:
                                msg = ask_grok(txt)
                        else:
                            msg = ask_grok(txt)

                    x_client.create_tweet(text=msg[:240], in_reply_to_tweet_id=tid)
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
        await asyncio.sleep(110)

async def main():
    await asyncio.gather(
        mention_loop(),
        hourly_post_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
