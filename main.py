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
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?search={}"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
mentions_timestamps = deque()
tweet_timestamps = deque()

# === AI Helpers ===
def ask_perplexity(prompt):
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "sonar-medium-online",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "temperature": 0.9
    }
    try:
        res = requests.post(PERPLEXITY_URL, json=body, headers=headers, timeout=35)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity fallback: {e}")
        return ask_grok(prompt)

def ask_grok(prompt):
    try:
        history_key = f"{REDIS_PREFIX}grok_history"
        body = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You're a bold, edgy crypto shill. Use only one fact from the knowledgebase."},
                {"role": "user", "content": prompt + "\n" + DEGEN_KNOWLEDGE + "\nEnd with NFA."}
            ],
            "max_tokens": 200,
            "temperature": 0.95
        }
        headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
        r = requests.post(GROK_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except:
        return "$DEGEN. NFA."

# === Token & Data ===
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
        return {
            'symbol': base.get('symbol'),
            'price_usd': float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
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
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\nMC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%\n{d['link']}"
    )

# === Posting ===
def post_raid(tweet):
    txt = tweet.text.lower()
    prompt = f"Write a bold one-liner hype for $DEGEN based on this: '{txt}'. Don't use the word 'raid'. Mention @ogdegenonsol. End with NFA."
    msg = ask_perplexity(prompt)
    img_list = glob.glob("raid_images/*.jpg")
    media = x_api.media_upload(choice(img_list)) if img_list else None
    x_client.create_tweet(
        text=msg[:240],
        in_reply_to_tweet_id=tweet.id,
        media_ids=[media.media_id_string] if media else None
    )
    db.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
    logger.info("Raid replied.")

async def mention_loop():
    while True:
        try:
            last_id = db.get(f"{REDIS_PREFIX}last_mention_id")
            res = x_client.get_users_mentions(id=BOT_ID, since_id=last_id, tweet_fields=['id', 'text'], max_results=10)
            if res.data:
                for tweet in reversed(res.data):
                    tid = tweet.id
                    if db.sismember(f"{REDIS_PREFIX}replied_ids", str(tid)):
                        continue
                    txt = tweet.text.replace('@askdegen', '').strip()
                    db.set(f"{REDIS_PREFIX}last_mention_id", tid)
                    db.sadd(f"{REDIS_PREFIX}replied_ids", str(tid))

                    if 'raid' in tweet.text.lower():
                        post_raid(tweet)
                        continue

                    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
                    if token:
                        sym, addr = resolve_token(token)
                        if addr:
                            data = fetch_data(addr)
                            msg = ask_perplexity(f"Analyze token {sym} based on this data: {json.dumps(data)}") if sym != 'DEGEN' else ask_perplexity(f"Shill $DEGEN hard using: {json.dumps(data)}")
                        else:
                            msg = ask_perplexity(txt)
                    elif txt.upper() == 'DEX':
                        data = fetch_data(DEGEN_ADDR)
                        msg = format_metrics(data)
                    elif txt.upper() == 'CA':
                        msg = f"Contract Address: {DEGEN_ADDR}"
                    else:
                        msg = ask_perplexity(txt)

                    x_client.create_tweet(text=msg[:240], in_reply_to_tweet_id=tid)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(d)
            prompt = "Give a short bullish update about $DEGEN in the last hour using context. Keep it punchy and varied."
            tweet = ask_perplexity(prompt)
            final = f"{metrics}\n\n{tweet}"
            last_post = db.get(f"{REDIS_PREFIX}last_hourly_post")
            if final.strip() != last_post:
                x_client.create_tweet(text=final[:560])
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
