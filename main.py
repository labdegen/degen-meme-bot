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
    "GROK_API_KEY", "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# API endpoints
GROK_URL = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?search={}"

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
        return "Not sure.  Just stack more Degen."

def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    payload = {
        'model': 'sonar-pro',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': 1.0,
        'top_p': 0.9,
        'search_recency_filter': 'week'
    }
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity fallback: {e}")
        return ask_grok("You are a professional market analyst.", user_prompt, max_tokens)

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
        'link':       f"https://dexscreener.com/solana/{addr}",
        'address':    addr
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
        resp = requests.get(SEARCH_URL.format(s), timeout=10)
        resp.raise_for_status()
        results = resp.json()
        for item in results:
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                symbol = base.get('symbol')
                addr = item.get('pairAddress') or base.get('address')
                if addr:
                    return symbol, addr
    except Exception as e:
        logger.warning(f"Dexscreener search failed for {s}: {e}")

    try:
        prompt = f"Find the Solana token contract address for the symbol {s}. Return JSON {{'symbol': str, 'address': str}}."
        out = ask_perplexity("You are a Solana token resolver.", prompt)
        data = json.loads(out)
        return data.get("symbol"), data.get("address")
    except Exception as e:
        logger.warning(f"Perplexity + Grok fallback for {s} failed: {e}")
        return None, None

def format_metrics(data: dict) -> str:
    return f"🚀 {data['symbol']} | ${data['price_usd']:,.6f}\nMC ${data['market_cap']:,.0f} | Vol24 ${data['volume_usd']:,.0f}\n1h {'🟢' if data['change_1h'] >= 0 else '🔴'}{data['change_1h']:+.2f}% | 24h {'🟢' if data['change_24h'] >= 0 else '🔴'}{data['change_24h']:+.2f}%\n{data['link']}"

def format_convo_reply(data: dict, question: str) -> str:
    prompt = f"Here are the token metrics: Symbol: {data['symbol']}, Price: ${data['price_usd']:.6f}, Market Cap: ${data['market_cap']:,.0f}, 24h Change: {data['change_24h']:+.2f}%, Volume: ${data['volume_usd']:,.0f}. Based on that, answer this in 230 characters or less, ending with 'NFA': {question}"
    return ask_grok("You are a crypto analyst replying to questions.", prompt, 120)

@app.get("/")
async def root():
    return {"status": "Degen bot is live."}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(poll_loop())
    asyncio.create_task(hourly_post_loop())

async def poll_loop():
    while True:
        last = db.get(f"{REDIS_PREFIX}last_tweet_id")
        since_id = int(last) if last else None
        res = x_client.get_users_mentions(
            id=BOT_ID,
            since_id=since_id,
            tweet_fields=['id', 'text', 'author_id'],
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
        await asyncio.sleep(90)

async def hourly_post_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            card = format_metrics(d)
            context = ask_grok("You're a Degen community member summarizing recent metrics. Make it casual, grounded, and complete within 2 sentences.", json.dumps(d), max_tokens=160)
            tweet = f"{card}\n{context}"
            if len(tweet) > 380:
                tweet = tweet[:380].rsplit('.', 1)[0] + '.'
            x_client.create_tweet(text=tweet)
            logger.info("Hourly promo posted")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
        await asyncio.sleep(3600)

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
                reply = format_metrics(d)
            else:
                reply = format_convo_reply(d, txt)
        else:
            reply = ask_perplexity("You are a crypto researcher. Answer this tweet in under 240 characters clearly.", txt, max_tokens=160)
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
