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
from tenacity import retry, stop_after_attempt, wait_exponential

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
    r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    """Enhanced Perplexity API call with retry logic"""
    payload = {
        'model': 'sonar-pro',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt or ''}
        ],
        'max_tokens': max_tokens,
        'temperature': 0.9,  # Slightly reduced for more consistent outputs
        'top_p': 0.9,
        'search_recency_filter': 'week'
    }
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        r = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Perplexity API error: {e}")
        raise


def fetch_data(addr: str) -> dict:
    cache_key = f"{REDIS_PREFIX}dex:{addr}"
    if cached := db.get(cache_key):
        return json.loads(cached)
    r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
    r.raise_for_status()
    data = r.json().get('pairs', [])[0] if r.json().get('pairs') else r.json()[0]
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
        data = resp.json()
        pairs = data.get('pairs', data)  # Handle both response formats
        for item in pairs:
            if item.get('chainId') == 'solana':
                base = item['baseToken']
                symbol = base.get('symbol')
                addr = item.get('pairAddress') or base.get('address')
                return symbol, addr
    except Exception as e:
        logger.warning(f"DexScreener search error: {e}")
    
    try:
        out = ask_grok(
            "Map Solana symbol to its contract address. Return JSON {'symbol':str,'address':str}.",
            f"Symbol: {s}",
            100
        )
        j = json.loads(out)
        return j.get('symbol'), j.get('address')
    except Exception as e:
        logger.warning(f"Grok token resolution error: {e}")
        return None, None


def clean_mention_text(text: str) -> str:
    """Better processing of mention text to handle multiple @mentions"""
    # Remove all @username mentions (not just @askdegen)
    cleaned = re.sub(r'@\w+', '', text).strip()
    return cleaned


async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or 'text' not in events[0]:
        logger.warning("Skipping empty mention payload")
        return {"message": "no events"}

    tweet_data = events[0]
    original_text = tweet_data['text']
    txt = clean_mention_text(original_text)
    tid = tweet_data['id_str']
    
    logger.info(f"Processing mention: {txt[:50]}...")

    # find first $TOKEN or address
    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    if token:
        sym, addr = resolve_token(token)
        if addr:
            d = fetch_data(addr)
            if txt.strip() == token:
                # pure metrics card
                lines = [
                    f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                    f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                    f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                    d['link']
                ]
                reply = "\n".join(lines)
            else:
                # metrics + conversational
                try:
                    prompt = f"Expert Solana analyst: {d['symbol']} metrics: price ${d['price_usd']:,.6f}, 24h change {d['change_24h']:+.2f}%, market cap ${d['market_cap']:,.0f}K. Reply conversationally to '{txt}' (<230 chars)."
                    reply = ask_perplexity(prompt, txt, max_tokens=100)
                except Exception:
                    # Fallback to more reliable but simpler response
                    prompt = f"As crypto analyst, comment briefly on {d['symbol']} price ${d['price_usd']:,.6f} with {d['change_24h']:+.2f}% 24h change."
                    reply = ask_grok(prompt, txt, max_tokens=100)
        else:
            try:
                reply = ask_perplexity("Crypto advisor: respond briefly to this query (<220 chars).", txt, max_tokens=100)
            except Exception:
                reply = ask_grok("Professional crypto professor: concise analytical response.", txt, max_tokens=100)
    else:
        try:
            reply = ask_perplexity("Expert crypto analyst: provide brief, helpful response (<220 chars).", txt, max_tokens=100)
        except Exception:
            reply = ask_grok("Professional crypto professor: concise analytical response.", txt, max_tokens=100)

    # ensure tweet is under character limit
    tweet = reply[:240]
    
    try:
        logger.info(f"Replying to tweet ID {tid} with: {tweet[:50]}...")
        x_client.create_tweet(text=tweet, in_reply_to_tweet_id=int(tid))
        logger.info("Reply posted successfully")
    except Exception as e:
        logger.error(f"Tweet reply error: {e}")
        try:
            # Legacy fallback
            x_api.update_status(status=tweet, in_reply_to_status_id=int(tid))
            logger.info("Reply posted via legacy API")
        except Exception as e2:
            logger.error(f"Legacy reply also failed: {e2}")
    
    return {'message': 'ok'}


async def degen_hourly_loop():
    """
    Hourly promo: DexScreener card + short analysis (<=280 chars)
    """
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            
            # Calculate remaining character space (Twitter limit is 280)
            card_length = sum(len(line) + 1 for line in card) - 1  # -1 because the last line doesn't need a newline
            remaining_chars = 280 - card_length - 1  # -1 for the newline between card and analysis
            
            # Adjust token limit to ensure we get an appropriate length response
            token_limit = max(20, remaining_chars // 4)  # Approx 4 chars per token as a safe estimate
            
            sys_msg = f"Professional crypto analyst: create a brief, positive yet realistic comment on $DEGEN (must be under {remaining_chars} characters)."
            
            try:
                analysis = ask_perplexity(sys_msg, "", max_tokens=token_limit)
                # Double-check length
                if len(analysis) > remaining_chars:
                    analysis = analysis[:remaining_chars]
            except Exception as e:
                logger.error(f"Perplexity promo error {e}, falling back to Grok")
                analysis = ask_grok(sys_msg, "", max_tokens=token_limit)
                if len(analysis) > remaining_chars:
                    analysis = analysis[:remaining_chars]
            
            tweet = "\n".join(card + [analysis])
            logger.info(f"Posting hourly promo ({len(tweet)}/280 chars)")
            
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted via v2")
            except Exception as e:
                logger.error(f"Twitter v2 API error: {e}")
                x_api.update_status(tweet)
                logger.info("Hourly promo posted via v1 fallback")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
        await asyncio.sleep(3600)


async def poll_loop():
    while True:
        try:
            last = db.get(f"{REDIS_PREFIX}last_tweet_id")
            since_id = int(last) if last else None
            
            logger.info(f"Polling mentions since ID: {since_id}")
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
                logger.info(f"Found {len(res.data)} new mentions")
                
                for tw in reversed(res.data):
                    ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                    try:
                        await handle_mention(ev)
                    except Exception as e:
                        logger.error(f"Mention handling error: {e}")
                    
                    db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                    db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
            
            # Adapt polling interval based on activity
            lm = db.get(f"{REDIS_PREFIX}last_mention")
            # Poll more frequently if there's been recent activity
            polling_interval = 90 if lm and time.time() - int(lm) < 3600 else 1800
            logger.info(f"Next poll in {polling_interval} seconds")
            await asyncio.sleep(polling_interval)
            
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
            await asyncio.sleep(300)  # Back off on errors


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