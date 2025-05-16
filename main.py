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
    "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
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
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
REDIS_CACHE_PREFIX = "degen:"
DEGEN_ADDRESS = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"

# Helper: call Grok API
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
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

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
        "symbol": data["baseToken"]["symbol"],
        "price_usd": float(data.get("priceUsd", 0)),
        "liquidity_usd": float(data.get("liquidity", {}).get("usd", 0)),
        "volume_usd": float(data.get("volume", {}).get("h24", 0)),
        "market_cap": float(data.get("marketCap", 0))
    }
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result

# Search recent tweets containing a symbol to provide context for address resolution
def search_symbol_context(symbol: str) -> str:
    query = f"${symbol} lang:en -is:retweet"
    tweets = x_client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text"])
    texts = " ".join([t.text for t in (tweets.data or [])])
    return texts[:2000]

# Resolve token symbol to address or validate address
def resolve_token(query: str) -> tuple:
    q = query.strip().upper()
    if q in ["DEGEN", "$DEGEN"]:
        return "DEGEN", DEGEN_ADDRESS
    # If looks like address
    if re.match(r"^[A-Za-z0-9]{43,44}$", q):
        return None, q
    # Else first try direct Grok map
    system = (
        "Crypto analyst: map a Solana token symbol to its smart contract address. "
        "Return JSON: {'symbol': str, 'address': str}."
    )
    user = f"Symbol: {q}"
    out = ask_grok(system, user, max_tokens=60)
    try:
        obj = json.loads(out)
        if obj.get("address"):
            return obj.get("symbol"), obj.get("address")
    except:
        pass
    # Fallback: survey recent tweets for context
    context = search_symbol_context(q)
    prompt = (
        "Based on these tweets: " + context +
        f" Find the most likely contract address for token {q}. Return JSON {{'symbol': str, 'address': str}}."
    )
    out2 = ask_grok(prompt, "Provide address mapping.", max_tokens=60)
    try:
        obj2 = json.loads(out2)
        return obj2.get("symbol"), obj2.get("address")
    except:
        return None, None

# Handle a mention event
async def handle_mention(data: dict):
    evt = data["tweet_create_events"][0]
    txt = evt.get("text", "").replace("@askdegen", "").strip()
    tid = evt.get("id_str")
    
    # Case 1: Degen Confession
    if txt.lower().startswith("degen confession:"):
        reply = ask_grok(
            "Witty degen summarizer: anonymize and condense confession ≤750 chars.",
            txt,
            max_tokens=200
        )
    # Case 2: $TOKEN or on-chain address
    elif txt.startswith("$") or re.search(r"^[A-Za-z0-9]{43,44}$", txt):
        token, address = resolve_token(txt.split()[0])
        if not address:
            reply = "Could not resolve token address."
        else:
            market_data = fetch_dexscreener_data(address)
            system = (


                "Expert crypto analyst specializing in Solana meme coins. Incorporate real-time sentiment and market data "
                f"{json.dumps(market_data)}. Craft a concise (≤380 chars) analysis of price action & sentiment. Focus on unique trends from large Solana accounts on X. Avoid mentioning $DOGE, $SHIB, $PEPE.  Insights should be remarkable and only possible from knowing the latest X data."
            )
            reply = ask_grok(system, txt, max_tokens=150)
    # Case 3: Freeform via Grok
    else:
        reply = ask_grok(
            "AtlasAI Degen Bot: professional assistant with dry degen humor—answer conversationally using real-time X data. Not goofy.  Smart and insightful.",
            txt,
            max_tokens=150
        )

    # Tweet the reply
    x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
    logger.info(f"Replied to {tid}: {reply}")
    return {"message": "Success"}, 200

# Polling loop to check mentions
async def poll_mentions():
    last_id = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    since_id = int(last_id) if last_id else None
    tweets = x_client.get_users_mentions(
        id=ASKDEGEN_ID,
        since_id=since_id,
        tweet_fields=["id", "text", "author_id"],
        expansions=["author_id"],
        user_fields=["username"],
        max_results=10
    )

    if not tweets or not tweets.data:
        return
    users = {u.id: u.username for u in tweets.includes.get("users", [])}
    for tw in reversed(tweets.data):
        user = users.get(tw.author_id, "unknown")
        event = {"tweet_create_events": [{"id_str": str(tw.id), "text": tw.text, "user": {"screen_name": user}}]}
        try:
            await handle_mention(event)
        except Exception as e:
            logger.error(f"Error handling mention: {e}")
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", tw.id)
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention", int(time.time()))

async def poll_mentions_loop():
    while True:
        await poll_mentions()
        lm = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        interval = 90 if (lm and time.time() - int(lm) < 3600) else 1800
        logger.info(f"Sleeping {interval}s until next poll")
        await asyncio.sleep(interval)

async def reset_daily_counters():
    key = f"{REDIS_CACHE_PREFIX}last_reset"
    last = redis_client.get(key)
    now = time.time()
    target = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
    if not last or int(last) < target:
        redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
        redis_client.set(f"{REDIS_CACHE_PREFIX}post_count", 0)
        redis_client.set(key, int(now))

@app.on_event("startup")
async def startup_event():
    await reset_daily_counters()
    asyncio.create_task(poll_mentions_loop())

@app.get("/")
async def root():
    return {"message": "Degen Meme Bot is live."}

@app.post("/test")
async def test_bot(request: Request):
    body = await request.json()
    txt = body.get("text", "@askdegen hello")
    event = {"tweet_create_events": [{"id_str": "0", "text": txt, "user": {"screen_name": "test"}}]}
    return await handle_mention(event)
