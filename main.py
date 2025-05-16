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
    "GROK_API_KEY", "PERPLEXITY_API_KEY",
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

# API endpoints
grok_url = "https://api.x.ai/v1/chat/completions"
perplexity_url = "https://api.perplexity.ai/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

# API keys
grok_key = os.getenv("GROK_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

# Redis setup
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

# Initialize X client
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

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDRESS_REGEX = re.compile(r'^[A-Za-z0-9]{43,44}$')

# Knowledge base for $DEGEN
DEGEN_KB = [
    "🚀 First $DEGEN on pump.fun (March 2024)",
    "🤝 100% organic community driven token",
    "🎮 Play Jeets vs Degens at jeetsvsdegens.com"
]

# Helpers
def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": max_tokens, "temperature": 0.7}
    headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
    resp = requests.post(grok_url, json=body, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    payload = {'model': 'sonar-pro', 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt or ''}], 'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'search_recency_filter': 'week'}
    headers = {'Authorization': f'Bearer {perplexity_key}', 'Content-Type': 'application/json'}
    resp = requests.post(perplexity_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

def fetch_dexscreener_data(addr: str) -> dict:
    cache_key = f"{REDIS_PREFIX}dex:{addr}"
    if cached := redis_client.get(cache_key):
        return json.loads(cached)
    resp = requests.get(f"{DEXS_URL}{addr}", timeout=10)
    resp.raise_for_status()
    d = resp.json()[0]
    t = d.get('baseToken', {})
    out = {
        'symbol': t.get('symbol'),
        'price_usd': float(d.get('priceUsd', 0)),
        'volume_usd': float(d.get('volume', {}).get('h24', 0)),
        'market_cap': float(d.get('marketCap', 0)),
        'change_1h': float(d.get('priceChange', {}).get('h1', 0)),
        'change_24h': float(d.get('priceChange', {}).get('h24', 0)),
        'project_url': t.get('projectUrl') or '',
        'socials': t.get('socials', [])
    }
    redis_client.setex(cache_key, 300, json.dumps(out))
    return out

def format_socials(socials: list) -> list:
    return [f"{soc['name']}: {soc['url']}" for soc in socials if soc.get('name') and soc.get('url')]

# Token resolution helper (symbol -> address)
def resolve_token(q: str) -> tuple:
    s = q.upper().lstrip('$')
    if s == 'DEGEN':
        return 'DEGEN', DEGEN_ADDR
    if ADDRESS_REGEX.match(s):
        return None, s
    # Fallback: always use Solana search via dexscreener API
    try:
        resp = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}", timeout=10)
        resp.raise_for_status()
        for item in resp.json():
            if item.get('chainId') == 'solana':
                sym = item.get('baseToken', {}).get('symbol')
                addr = item.get('pairAddress') or item.get('baseToken', {}).get('address')
                return sym, addr
    except:
        pass
    # Last fallback via Grok
    out = ask_grok(
        'Map a Solana token symbol to its contract address. Return JSON {"symbol":str,"address":str}.',
        f"Symbol: {s}",
        100
    )
    try:
        j = json.loads(out)
        return j.get('symbol'), j.get('address')
    except:
        return None, None

('name') and soc.get('url')]

async def handle_mention(ev: dict):
    txt = ev['tweet_create_events'][0]['text'].replace('@askdegen', '').strip()
    tid = ev['tweet_create_events'][0]['id_str']
    tokens = [w for w in txt.split() if w.startswith('$') or ADDRESS_REGEX.match(w)]
    if tokens:
        q = tokens[0]
        tok, addr = resolve_token(q)
        if addr:
            d = fetch_dexscreener_data(addr)
            if txt.strip() == q:
                lines = [
                    f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                    f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                    f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"
                ]
                if d['project_url']:
                    lines.append(f"🌐 {d['project_url']}")
                lines += format_socials(d['socials'])
                lines.append(f"🔗 https://dexscreener.com/solana/{addr}")
                if tok == 'DEGEN': lines += DEGEN_KB
                reply = '\n'.join(lines)
            else:
                system = f"Expert Solana meme coin analyst: given these metrics {json.dumps(d)}, craft a concise (<240 chars) conversational reply."
                reply = ask_perplexity(system, txt, max_tokens=150)
        else:
            reply = ask_perplexity("Crypto details unavailable—one concise tweet.", txt, max_tokens=80)
    else:
        reply = ask_grok("Answer as Tim Dillon: witty, direct, one tweet (<240 chars).", txt, max_tokens=120)
    x_client.create_tweet(text=reply[:240], in_reply_to_tweet_id=int(tid))
    return {'message':'ok'}

async def degen_hourly_loop():
    """
    Every hour, fetch fresh Solana $DEGEN metrics and post a 4-sentence promo tweet.
    Always uses the Solana contract address DEGEN_ADDR.
    """
    while True:
        try:
            # Explicitly fetch data using the Solana contract address
            logger.info(f"Fetching Solana $DEGEN metrics for promo: {DEGEN_ADDR}")
            d = fetch_dexscreener_data(DEGEN_ADDR)
            # Compose the promo text via Perplexity, without including the address
            system = (
                "Dynamic promo copywriter: write exactly 4 sentences that are positive, engaging, and community-focused about $DEGEN on Solana, "
                f"using the latest metrics: price=${d['price_usd']:,.6f}, market cap=${d['market_cap']:,.0f}K, 24h volume=${d['volume_usd']:,.1f}K." 
                "Return only the tweet text, up to 280 characters."
            )
            promo = ask_perplexity(system, "", max_tokens=180)
            x_client.create_tweet(text=promo.strip()[:280])
            logger.info("Hourly promo sent successfully.")
        except Exception as e:
            logger.error(f"promo error: {e}")
        # Wait one hour before next promo
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        last = redis_client.get(f"{REDIS_PREFIX}last_tweet_id")
        since = int(last) if last else None
        res = x_client.get_users_mentions(id=ASKDEGEN_ID, since_id=since, tweet_fields=['id','text','author_id'], expansions=['author_id'], user_fields=['username'], max_results=10)
        if res and res.data:
            users = {u.id: u.username for u in res.includes.get('users', [])}
            for tw in reversed(res.data):
                ev = {'tweet_create_events':[{'id_str':str(tw.id),'text':tw.text,'user':{'screen_name':users.get(tw.author_id,'?')}}]}
                try: await handle_mention(ev)
                except Exception as e: logger.error(f"handle_mention error: {e}")
                redis_client.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                redis_client.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
        lm = redis_client.get(f"{REDIS_PREFIX}last_mention")
        await asyncio.sleep(90 if lm and time.time()-int(lm)<3600 else 1800)

@app.on_event('startup')
async def on_startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message':'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    b = await r.json()
    ev = {'tweet_create_events':[{'id_str':'0','text':b.get('text',''),'user':{'screen_name':'test'}}]}
    return await handle_mention(ev)
