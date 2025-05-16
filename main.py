from fastapi import FastAPI, HTTPException, Request
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load env vars and validate
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# API endpoints
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

# Credentials
API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
GROK_KEY = os.getenv("GROK_API_KEY")

# Redis client
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Initialize Tweepy
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

# Helper functions
def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {GROK_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Grok fallback: {e}")
        return "Insight currently unavailable. Check back soon."

def fetch_data(addr: str) -> dict:
    cache_key = f"{REDIS_PREFIX}dex:{addr}"
    if cached := db.get(cache_key):
        return json.loads(cached)
    r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
    r.raise_for_status()
    data = r.json()[0]
    base = data.get('baseToken', {})
    out = {
        'symbol':     base.get('symbol'),
        'price_usd':  float(data.get('priceUsd', 0)),
        'volume_usd': float(data.get('volume', {}).get('h24', 0)),
        'market_cap': float(data.get('marketCap', 0)),
        'change_1h':  float(data.get('priceChange', {}).get('h1', 0)),
        'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
        'link':       f"https://dexscreener.com/solana/{addr}"
    }
    db.setex(cache_key, 300, json.dumps(out))
    return out

def resolve_token(q: str) -> tuple:
    s = q.upper().lstrip('$')
    if s == 'DEGEN':
        return 'DEGEN', DEGEN_ADDR
    if ADDR_RE.match(s):
        return None, s
    try:
        resp = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}", timeout=10)
        resp.raise_for_status()
        for item in resp.json():
            if item.get('chainId') == 'solana':
                base = item['baseToken']
                symbol = base.get('symbol')
                addr = item.get('pairAddress') or base.get('address')
                return symbol, addr
    except:
        pass
    return None, None

async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or not isinstance(events, list) or not events[0].get("text"):
        logger.warning("Skipping invalid or empty mention event")
        return {"message": "no valid mention"}

    txt = events[0]['text'].replace('@askdegen', '').strip()
    tid = events[0]['id_str']

    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    if token:
        sym, addr = resolve_token(token)
        if addr:
            d = fetch_data(addr)
            if txt.strip() == token:
                lines = [
                    f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                    f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                    f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                    d['link']
                ]
                reply = "\n".join(lines)
            else:
                prompt = f"You are a professional crypto market analyst. Given: {json.dumps(d)}, reply to an investor in <240 characters. Be clear, insightful, and do not mention Solana."
                reply = ask_grok(prompt, txt, max_tokens=160)
        else:
            reply = ask_grok("Professional analyst: reply under 240 characters clearly.", txt, max_tokens=160)
    else:
        reply = ask_grok("Professional crypto professor: concise analytical response.", txt, max_tokens=160)

    tweet = reply.strip()
    if len(tweet) > 240:
        tweet = tweet[:240]
        if '.' in tweet:
            tweet = tweet.rsplit('.', 1)[0] + '.'
        else:
            tweet = tweet.rsplit(' ', 1)[0] + '...'
    x_client.create_tweet(text=tweet, in_reply_to_tweet_id=int(tid))
    return {'message': 'ok'}

async def degen_hourly_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            sys_msg = "You are a DEGEN community insider. Write a 2-sentence hourly update based on this data. Be enthusiastic but grounded. Do not mention Solana."
            analysis = ask_grok(sys_msg, json.dumps(d), max_tokens=160)
            tweet = "\n".join(card + [analysis])
            if len(tweet) > 280:
                tweet = tweet[:280]
                if '.' in tweet:
                    tweet = tweet.rsplit('.', 1)[0] + '.'
                else:
                    tweet = tweet.rsplit(' ', 1)[0] + '...'
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted")
            except:
                x_api.update_status(tweet)
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        last = db.get(f"{REDIS_PREFIX}last_tweet_id")
        since_id = int(last) if last else None
        res = x_client.get_users_mentions(
            id=BOT_ID,
            since_id=since_id,
            tweet_fields=['id','text','author_id'],
            expansions=['author_id'],
            user_fields=['username'],
            max_results=10
        )
        if res and res.data:
            users = {u.id: u.username for u in res.includes.get('users', [])}
            for tw in reversed(res.data):
                ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                try:
                    await handle_mention(ev)
                except Exception as e:
                    logger.error(f"Mention error: {e}")
                db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
        lm = db.get(f"{REDIS_PREFIX}last_mention")
        await asyncio.sleep(90 if lm and time.time() - int(lm) < 3600 else 1800)

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    data = await r.json()
    ev = {'tweet_create_events': [{'id_str': '0', 'text': data.get('text', ''), 'user': {'screen_name': 'test'}}]}
    return await handle_mention(ev)
