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
import random

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
    "GROK_API_KEY","PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Twitter (X) client setup
api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("X_BEARER_TOKEN")

grok_url = "https://api.x.ai/v1/chat/completions"

grok_key = os.getenv("GROK_API_KEY")
perplexity_url = "https://api.perplexity.ai/chat/completions"
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

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
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    r = requests.post(grok_url, json=body, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

x_client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)
# Get bot ID
token_data = x_client.get_me().data
ASKDEGEN_ID = token_data.id
logger.info(f"Authenticated as: {token_data.username}, ID: {ASKDEGEN_ID}")

# Redis client
a = os.getenv
redis_client = redis.Redis(
    host=a("REDIS_HOST"), port=int(a("REDIS_PORT")), password=a("REDIS_PASSWORD"),
    decode_responses=True, socket_timeout=5, socket_connect_timeout=5
)
redis_client.ping(); logger.info("Redis connected")

# Config
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
DEX_SEARCH = "https://api.dexscreener.com/latest/dex/search?search={}"
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
PROMO_IDX = f"{REDIS_PREFIX}promo_idx"

# Knowledgebase for $DEGEN responses
DEGEN_KB = [
    "🚀 1st $DEGEN ticker on pump.fun Debuted in March 2024.",
    "🤝 Fully organic community token.",
    "🎮 Play our game Jeets vs Degens at jeetsvsdegens.com"
]

# Fetch DexScreener data
def fetch_dexscreener_data(addr: str) -> dict:
    key = f"{REDIS_PREFIX}dex:{addr}"
    if v := redis_client.get(key): return json.loads(v)
    r = requests.get(f"{DEXS_URL}{addr}", timeout=10); r.raise_for_status()
    d = r.json()[0]
    out = {
        'symbol': d['baseToken']['symbol'],
        'price_usd': float(d.get('priceUsd',0)),
        'fdv_usd': float(d.get('fdv',0)),
        'volume_usd': float(d.get('volume',{}).get('h24',0)),
        'liquidity_usd': float(d.get('liquidity',{}).get('usd',0)),
        'market_cap': float(d.get('marketCap',0)),
        'change_1h': float(d.get('priceChange',{}).get('h1',0)),
        'change_24h': float(d.get('priceChange',{}).get('h24',0))
    }
    redis_client.setex(key,300,json.dumps(out)); return out

# Search by ticker

def search_symbol(sym: str):
    try:
        r = requests.get(DEX_SEARCH.format(sym), timeout=10); r.raise_for_status()
        for it in r.json():
            if it.get('chainId')=='solana':
                return it['baseToken']['symbol'], it.get('pairAddress') or it['baseToken']['address']
    except: pass
    return None,None

# Resolve token
def resolve_token(q: str):
    s = q.upper().lstrip('$')
    if s=='DEGEN': return 'DEGEN', DEGEN_ADDR
    if re.match(r'^[A-Za-z0-9]{43,44}$', s): return None, s
    if sym,addr:=search_symbol(s): return sym,addr
    out = ask_grok("Map a Solana token symbol to its address. Return JSON {\"symbol\":str,\"address\":str}.", f"Symbol: {s}",100)
    try: j=json.loads(out); return j.get('symbol'), j.get('address')
    except: return None,None

# Handle mention
async def handle_mention(ev: dict):
    txt=ev['tweet_create_events'][0]['text'].replace('@askdegen','').strip(); tid=ev['tweet_create_events'][0]['id_str']
    w=txt.split()
    # Crypto query
    if len(w)==1 and (w[0].startswith('$') or re.match(r'^[A-Za-z0-9]{43,44}$',w[0])):
        tok,addr=resolve_token(w[0])
        if not addr:
            reply=ask_perplexity("Crypto data unavailable—sorry! Reply in one tweet under 240 chars.", txt,80)
        else:
            d=fetch_dexscreener_data(addr)
            lines=[
                f"🚀 {d['symbol']} | ${d['price_usd']:.6f}"
                ,f"MC ${d['market_cap']:.0f}K | 24h Vol ${d['volume_usd']:.1f}K"
                ,f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"
                ,f"🔗https://dexscreener.com/solana/{addr}"
            ]
            if tok=='DEGEN': lines+=DEGEN_KB
            reply="\n".join(lines)
        x_client.create_tweet(text=reply[:240],in_reply_to_tweet_id=int(tid)); return{'message':'ok'},200
    # Non-crypto
    reply=ask_grok("Answer as Tim Dillon—witty, direct, one tweet.", txt,120)
    x_client.create_tweet(text=reply[:240],in_reply_to_tweet_id=int(tid)); return{'message':'ok'},200

# Hourly promo loop with rotating templates
TEMPLATES=[
    "🚀 $DEGEN at ${price:.6f} and MC ${mcap:.0f}K—strong volume means real traction among Solana traders. @ogdegenonsol",
    "💥 $DEGEN trades at ${price:.6f} with MC ${mcap:.0f}K—healthy liquidity shows deep pool support. @ogdegenonsol",
    "🔥 $DEGEN: ${price:.6f}, MC ${mcap:.0f}K—community buzz hitting new highs. @ogdegenonsol",
    "🎉 $DEGEN now ${price:.6f}, MC ${mcap:.0f}K—organic growth driving momentum. @ogdegenonsol"
]

def compose_promo():
    idx=int(redis_client.get(PROMO_IDX)or0)%len(TEMPLATES)
    redis_client.set(PROMO_IDX,(idx+1)%len(TEMPLATES))
    d=fetch_dexscreener_data(DEGEN_ADDR)
    return TEMPLATES[idx].format(price=d['price_usd'],mcap=d['market_cap'])[:240]

async def degen_hourly_loop():
    while True:
        try: x_client.create_tweet(text=compose_promo()); logger.info('promo sent')
        except Exception as e: logger.error(f'promo err:{e}')
        await asyncio.sleep(3600)

# Polling & startup
async def poll_mentions():
    last=redis_client.get(f"{REDIS_PREFIX}last_tweet_id")
    since=int(last) if last else None
    res=x_client.get_users_mentions(id=ASKDEGEN_ID,since_id=since,tweet_fields=['id','text','author_id'],expansions=['author_id'],user_fields=['username'],max_results=10)
    if not res or not res.data: return
    users={u.id:u.username for u in res.includes.get('users',[])}
    for tw in reversed(res.data):
        ev={'tweet_create_events':[{'id_str':str(tw.id),'text':tw.text,'user':{'screen_name':users.get(tw.author_id,'?')}}]}
        try: await handle_mention(ev)
        except Exception as e: logger.error(e)
        redis_client.set(f"{REDIS_PREFIX}last_tweet_id",tw.id)
        redis_client.set(f"{REDIS_PREFIX}last_mention",int(time.time()))

async def poll_loop():
    while True:
        await poll_mentions(); lm=redis_client.get(f"{REDIS_PREFIX}last_mention");
        await asyncio.sleep(90 if lm and time.time()-int(lm)<3600 else 1800)

@app.on_event('startup')
async def on_startup():
    asyncio.create_task(poll_loop()); asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root(): return {'message':'live.'}

@app.post('/test')
async def test_bot(r:Request): b=await r.json();return await handle_mention({'tweet_create_events':[{'id_str':'0','text':b.get('text',''),'user':{'screen_name':'test'}}]})
