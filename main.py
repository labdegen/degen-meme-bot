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
tlogging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Set up X (Twitter) client\api_key             = os.getenv("X_API_KEY")
api_key_secret      = os.getenv("X_API_KEY_SECRET")
access_token        = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token        = os.getenv("X_BEARER_TOKEN")

x_client = tweepy.Client(
    bearer_token        = bearer_token,
    consumer_key        = api_key,
    consumer_secret     = api_key_secret,
    access_token        = access_token,
    access_token_secret = access_token_secret,
)

# Redis client
redis_client = redis.Redis(
    host                   = os.getenv("REDIS_HOST"),
    port                   = int(os.getenv("REDIS_PORT")),
    password               = os.getenv("REDIS_PASSWORD"),
    decode_responses       = True,
    socket_timeout         = 5,
    socket_connect_timeout = 5
)
redis_client.ping()
logger.info("Redis connected")

# Config
GROK_URL           = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY       = os.getenv("GROK_API_KEY")
DEXSCREENER_URL    = "https://api.dexscreener.com/token-pairs/v1/solana/"
REDIS_CACHE_PREFIX = "degen:"
DEGEN_ADDRESS      = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"

# Helper to call Grok
```python
```
def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
    return json.loads(r.json()["choices"][0]["message"]["content"].strip())

# Fetch live DexScreener data
def fetch_dexscreener_data(address: str) -> dict:
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except:
        pass

    r = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
    r.raise_for_status()
    data = r.json()[0]
    result = {
        "symbol": data["baseToken"]["symbol"],
        "price_usd": float(data.get("priceUsd", 0)),
        "liquidity_usd": float(data.get("liquidity", {}).get("usd", 0)),
        "volume_usd": float(data.get("volume", {}).get("h24", 0)),
        "market_cap": float(data.get("marketCap", 0))
    }
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result

# Resolve a token symbol to contract address or validate address
address_regex = re.compile(r"^[A-Za-z0-9]{43,44}$")
def resolve_token(query: str) -> tuple:
    q = query.strip().upper()
    # Degen special
    if q in ["DEGEN", "$DEGEN"]:
        return "DEGEN", DEGEN_ADDRESS
    # If looks like address
    if address_regex.match(q):
        return None, q
    # Otherwise ask Grok to map symbol → address
    system = "Crypto analyst: map a Solana token symbol to its smart contract address. Return JSON: {'symbol': str, 'address': str}."
    user   = f"Symbol: {q}"
    out = ask_grok(system, user, max_tokens=60)
    return out.get("symbol"), out.get("address")

# Handle incoming mention\async def handle_mention(data: dict):
    evt = data["tweet_create_events"][0]
    txt = evt["text"].replace("@askdegen", "").strip()
    tid = evt["id_str"]

    # 1) Degen confession
    if txt.lower().startswith("degen confession:"):
        reply = ask_grok(
            "Witty degen bot summarizer: condense and anonymize confession ≤750 chars.",
            txt,
            max_tokens=200
        )

    # 2) $TOKEN or address lookup
    elif txt.startswith("$") or address_regex.search(txt):
        symbol, address = resolve_token(txt.split()[0])
        data = fetch_dexscreener_data(address)
        system = (
            "Dry crypto gambler analyst. Given market data "
            f"{json.dumps(data)}, craft a concise (≤280 chars) degen‐toned tweet about current price action & sentiment."
        )
        reply = ask_grok(system, txt, max_tokens=150)

    # 3) Freeform queries via Grok
    else:
        reply = ask_grok(
            "AtlasAI Degen Bot: professional assistant with dry degen humor—answer conversationally.",
            txt,
            max_tokens=150
        )

    # Send the tweet using OAuth 1.0a reply
    x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
    logger.info(f"Replied to {tid}: {reply}")
    return {"message": "Success"}, 200

# Polling and startup boilerplate
async def poll_mentions():
    last_id = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    since_id = int(last_id) if last_id else None

    tweets = x_client.get_users_mentions(
        id=ASKDEGEN_ID,
        since_id=since_id,
        tweet_fields=["id","text","author_id"],
        expansions=["author_id"],
        user_fields=["username"],
        max_results=10
    )

    if tweets.data:
        users = {u.id: u.username for u in tweets.includes.get("users", [])}
        for tw in reversed(tweets.data):
            user = users.get(tw.author_id, "unknown")
            event = {"tweet_create_events":[{"id_str":str(tw.id),"text":tw.text,"user":{"screen_name":user}}]}
            await handle_mention(event)
            redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", tw.id)
            redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention", int(time.time()))

async def poll_mentions_loop():
    while True:
        await poll_mentions()
        lm = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        sleep_time = 90 if (lm and time.time()-int(lm)<3600) else 1800
        logger.info(f"Sleeping {sleep_time}s until next poll")
        await asyncio.sleep(sleep_time)

async def reset_daily_counters():
    key = f"{REDIS_CACHE_PREFIX}last_reset"
    last = redis_client.get(key)
    now = time.time()
    target = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00","%Y-%m-%d %H:%M:%S")) - (4*3600)
    if not last or int(last)<target:
        redis_client.set(f"{REDIS_CACHE_PREFIX}read_count",0)
        redis_client.set(f"{REDIS_CACHE_PREFIX}post_count",0)
        redis_client.set(key,int(now))

@app.on_event("startup")
async def start():
    await reset_daily_counters()
    asyncio.create_task(poll_mentions_loop())

@app.get("/")
async def root():
    return {"message":"Degen Meme Bot is live."}

@app.post("/test")
async def test_bot(req: Request):
    body = await req.json()
    txt = body.get("text","@askdegen hello")
    evt = {"tweet_create_events":[{"id_str":"000","text":txt,"user":{"screen_name":"test"}}]}
    return await handle_mention(evt)
