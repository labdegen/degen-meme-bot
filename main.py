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

# Load environment variables
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "PERPLEXITY_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Set up Twitter (X) client
api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("X_BEARER_TOKEN")

x_client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
)
# Get own ID
token_data = x_client.get_me().data
ASKDEGEN_ID = token_data.id
logger.info(f"Authenticated as: {token_data.username}, ID: {ASKDEGEN_ID}")

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5
)
redis_client.ping()
logger.info("Redis connected")

# Config
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
REDIS_CACHE_PREFIX = "degen:"
DEGEN_ADDRESS = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"

# Helper: call Perplexity API
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
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json'
    }
    resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

# Fetch live data from DexScreener
def fetch_dexscreener_data(address: str) -> dict:
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except:
        pass
    resp = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
    resp.raise_for_status()
    data = resp.json()[0]
    result = {
        'symbol': data['baseToken']['symbol'],
        'price_usd': float(data.get('priceUsd', 0)),
        'liquidity_usd': float(data.get('liquidity', {}).get('usd', 0)),
        'volume_usd': float(data.get('volume', {}).get('h24', 0)),
        'market_cap': float(data.get('marketCap', 0))
    }
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result

# Resolve token symbol or address
def resolve_token(query: str) -> tuple:
    q = query.strip().upper()
    if q in ['DEGEN', '$DEGEN']:
        return 'DEGEN', DEGEN_ADDRESS
    if re.match(r'^[A-Za-z0-9]{43,44}$', q):
        return None, q
    # Try direct Perplexity mapping
    sys = 'Map a Solana token symbol to its contract address. Return JSON {"symbol":str,"address":str}.'
    out = ask_perplexity(sys, f"Symbol: {q}", max_tokens=100)
    try:
        obj = json.loads(out)
        if obj.get('address'):
            return obj.get('symbol'), obj.get('address')
    except:
        pass
    return None, None

# Handle a mention event
async def handle_mention(data: dict):
    evt = data['tweet_create_events'][0]
    txt = evt.get('text','').replace('@askdegen','').strip()
    tid = evt.get('id_str')

    # If pure token/address query
    words = txt.split()
    is_symbol = len(words)==1 and (words[0].startswith('$') or re.match(r'^[A-Za-z0-9]{43,44}$',words[0]))
    if is_symbol:
        _, address = resolve_token(words[0])
        if not address:
            reply = 'Could not resolve token address.'
        else:
            d = fetch_dexscreener_data(address)
            reply = (
                f"{d['symbol']} | Price: ${d['price_usd']:.6f} | "
                f"24h Vol: ${d['volume_usd']:.2f} | Liquidity: ${d['liquidity_usd']:.2f} | "
                f"MC: ${d['market_cap']:.2f}"
                f"\n🔗 https://dexscreener.com/solana/{address}"
            )
        # single tweet
        x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
        return {'message':'Success'},200

    # Other queries -> Perplexity
    reply = ask_perplexity(
        'Professional degen assistant: answer concisely in one tweet (<280 chars).',
        txt,
        max_tokens=120
    )
    x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
    return {'message':'Success'},200

# Hourly DEGEN promotional loop
async def degen_hourly_loop():
    while True:
        d = fetch_dexscreener_data(DEGEN_ADDRESS)
        tweet = (
            f"${d['symbol']} is trading at ${d['price_usd']:.6f} with a market cap of ${d['market_cap']:.2f}. "
            f"Upbeat momentum in Solana meme space—now's a prime entry under ATH!"
        )
        x_client.create_tweet(text=tweet)
        logger.info('Posted hourly DEGEN update')
        await asyncio.sleep(3600)

@app.on_event('startup')
async def startup_event():
    asyncio.create_task(poll_mentions_loop())
    asyncio.create_task(degen_hourly_loop())

# Polling loop (unchanged)
async def poll_mentions():
    last = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    since = int(last) if last else None
    res = x_client.get_users_mentions(
        id=ASKDEGEN_ID, since_id=since,
        tweet_fields=['id','text','author_id'],
        expansions=['author_id'], user_fields=['username'],
        max_results=10
    )
    if not res or not res.data:
        return
    users = {u.id:u.username for u in res.includes.get('users',[])}
    for tw in reversed(res.data):
        evt = {'tweet_create_events':[{'id_str':str(tw.id),'text':tw.text,'user':{'screen_name':users.get(tw.author_id,'?')}}]}
        try: await handle_mention(evt)
        except Exception as e: logger.error(e)
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id",tw.id)
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention",int(time.time()))

async def poll_mentions_loop():
    while True:
        await poll_mentions()
        lm=redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        interval = 90 if lm and time.time()-int(lm)<3600 else 1800
        await asyncio.sleep(interval)

@app.get('/')
async def root(): return {'message':'Degen Meme Bot is live.'}
@app.post('/test')
async def test_bot(r:Request): body=await r.json();txt=body.get('text','');evt={'tweet_create_events':[{'id_str':'0','text':txt,'user':{'screen_name':'test'}}]};return await handle_mention(evt)
