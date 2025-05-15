from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import json
from datetime import datetime, timedelta
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing environment variable: {var}")
        raise RuntimeError(f"Missing environment variable: {var}")

# Tweepy client (@askdegen creds)
try:
    x_client = tweepy.Client(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    user = x_client.get_me()
    logger.info(f"X API authenticated as: {user.data.username}")
except tweepy.TweepyException as e:
    logger.error(f"X API authentication failed: {str(e)}")
    raise RuntimeError(f"X API authentication failed: {str(e)}")

# Redis client
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info("Redis connection successful")
except redis.RedisError as e:
    logger.error(f"Redis connection failed: {str(e)}")
    raise RuntimeError(f"Redis connection failed: {str(e)}")

# Grok and DexScreener config
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?q="
REDIS_CACHE_PREFIX = "degen:"

# Token regex for $TOKEN or contract address
HEX_REGEX = re.compile(r"^[A-Za-z0-9]{43,44}$")

def fetch_dexscreener_data(address: str, retries=3, backoff=2) -> dict:
    """Fetch token metrics from DexScreener with caching."""
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for DexScreener: {address}")
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis cache get failed: {str(e)}")

    for attempt in range(retries):
        try:
            response = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or not isinstance(data, list) or not data[0]:
                logger.warning(f"No DexScreener data for {address}")
                return {}
            pair = data[0]
            result = {
                "token_name": pair.get("baseToken", {}).get("name", "Unknown"),
                "token_symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
                "price_usd": float(pair.get("priceUsd", 0)),
                "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                "volume_usd": float(pair.get("volume", {}).get("h24", 0)),
                "txns": pair.get("txns", {}).get("h24", {}).get("buys", 0) + pair.get("txns", {}).get("h24", {}).get("sells", 0),
                "market_cap": float(pair.get("marketCap", 0))
            }
            try:
                redis_client.setex(cache_key, 300, json.dumps(result))
            except redis.RedisError as e:
                logger.error(f"Redis cache set failed: {str(e)}")
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                logger.warning(f"DexScreener rate limit, retrying in {backoff}s...")
                sleep(backoff)
                backoff *= 2
                continue
            logger.error(f"DexScreener error for {address}: {str(e)}")
            return {}
        except requests.RequestException as e:
            logger.error(f"DexScreener error for {address}: {str(e)}")
            return {}
    return {}

def resolve_token(query: str) -> tuple:
    """Resolve query to $TOKEN or contract address using X sentiment."""
    query = query.strip().lower()
    is_contract = HEX_REGEX.match(query)

    if is_contract:
        # Contract address: fetch DexScreener directly
        dexscreener_data = fetch_dexscreener_data(query)
        if dexscreener_data.get("token_symbol", "Unknown") != "Unknown":
            return dexscreener_data["token_symbol"].upper(), query, dexscreener_data
        # Fallback to X search for token name
        system = (
            "You are a crypto analyst. Given a Solana contract address, find the $TOKEN ticker from recent X posts. "
            "Return JSON: {'token': str, 'address': str}"
        )
        user_msg = f"Contract: {query}. Find the Solana $TOKEN ticker and confirm address from X posts."
    elif query.startswith("most hyped") or "meme coin" in query:
        # Meme coin query: find trending Solana meme coin
        system = (
            "You are a crypto analyst. Identify the most hyped Solana meme coin today from X posts, excluding $SOL, $USDC, $USDT. "
            "Return JSON: {'token': str, 'address': str}"
        )
        user_msg = "Find the Solana meme coin with the highest X buzz today, ignoring stablecoins and $SOL."
    else:
        # $TOKEN or plain ticker: search X for Solana contract address
        token = query.replace("$", "").upper()
        system = (
            "You are a crypto analyst. Given a $TOKEN ticker, find the most relevant Solana contract address from recent X posts. "
            "Return JSON: {'token': str, 'address': str}"
        )
        user_msg = f"Ticker: {token}. Find the Solana contract address from X posts."

    # Check Redis cache for resolved token
    cache_key = f"{REDIS_CACHE_PREFIX}resolve:{query}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for token resolution: {query}")
            data = json.loads(cached)
            token = data.get("token", "UNKNOWN").upper()
            address = data.get("address", "")
            dexscreener_data = fetch_dexscreener_data(address) if address else {}
            return token, address, dexscreener_data
    except redis.RedisError as e:
        logger.error(f"Redis cache get failed: {str(e)}")

    # Query Grok for token resolution
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        r.raise_for_status()
        response = r.json()
        text = response["choices"][0]["message"]["content"].strip()
        data = json.loads(text)
        token = data.get("token", "UNKNOWN").replace("$", "").upper()
        address = data.get("address", "")

        # Cache resolution
        try:
            redis_client.setex(cache_key, 3600, json.dumps({"token": token, "address": address}))
        except redis.RedisError as e:
            logger.error(f"Redis cache set failed: {str(e)}")

        # Fetch DexScreener data if address found
        dexscreener_data = fetch_dexscreener_data(address) if address else {}
        return token, address, dexscreener_data
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Token resolution failed for {query}: {str(e)}")
        return "UNKNOWN", "", {}

def analyze_query(query: str, token: str, address: str, dexscreener_data: dict, tid: str) -> dict:
    """Analyze query using Grok, blending X sentiment and DexScreener data."""
    cache_key = f"{REDIS_CACHE_PREFIX}analysis:{tid}:{query}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for analysis: {query}")
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis cache get failed: {str(e)}")

    # Check conversation context
    context_key = f"{REDIS_CACHE_PREFIX}context:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError as e:
        logger.error(f"Redis cache get failed: {str(e)}")
        prior_context = {"query": "", "response": ""}

    system = (
        "You are a cynical crypto analyst and all-knowing expert. For crypto queries, use X sentiment and market data to give sharp insights. "
        "Exclude $SOL, $USDC, $USDT for meme coin queries. For non-crypto (e.g., life advice, weather), provide witty, authoritative answers. "
        "Return JSON: {'reply': str, 'is_crypto': bool}, reply <200 chars."
    )
    user_msg = (
        f"Query: {query}. Token: {token}, Address: {address}. "
        f"Market: {json.dumps(dexscreener_data)}. "
        f"Prior: Query: {prior_context['query']}, Response: {prior_context['response']}. "
        "Give a cynical, actionable reply under 200 chars. For meme coins, focus on hype, risk. For non-crypto, be clever."
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        "max_tokens": 200,
        "temperature": 0.7
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        r.raise_for_status()
        response = r.json()
        text = response["choices"][0]["message"]["content"].strip()
        data = json.loads(text)
        reply = data.get("reply", "No edge here. Try $BONK or chill.")[:200]
        is_crypto = data.get("is_crypto", False)

        # Cache analysis and context
        try:
            redis_client.setex(cache_key, 300, json.dumps(data))
            redis_client.setex(context_key, 3600, json.dumps({"query": query, "response": reply}))
        except redis.RedisError as e:
            logger.error(f"Redis cache set failed: {str(e)}")

        return data
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Analysis failed for {query}: {str(e)}")
        return {"reply": "No edge here. Try $BONK or chill.", "is_crypto": False}

@app.post("/")
async def handle_mention(data: dict):
    """Handle X mentions for @askdegen."""
    logger.info(f"Received payload: {json.dumps(data, indent=2)}")
    try:
        if "tweet_create_events" not in data or not data["tweet_create_events"]:
            logger.error(f"Invalid payload: {data}")
            return JSONResponse({"message": "No tweet data"}, status_code=400)

        evt = data["tweet_create_events"][0]
        txt = evt.get("text", "").replace("@askdegen", "").strip()
        user = evt.get("user", {}).get("screen_name", "")
        tid = evt.get("id_str", "")
        in_reply_to_status_id = evt.get("in_reply_to_status_id_str", None)

        if not all([txt, user, tid]):
            logger.error(f"Missing tweet data: text={txt}, user={user}, tid={tid}")
            return JSONResponse({"message": "Invalid tweet data"}, status_code=400)

        logger.info(f"Processing tweet ID: {tid}, user: {user}, text: {txt}")

        # Extract query
        words = txt.split()
        query = None
        for w in words:
            if w.startswith("$") and len(w) > 1:
                query = w[1:]
                break
            if HEX_REGEX.match(w):
                query = w
                break
            if "meme coin" in w.lower() or "hyped token" in w.lower():
                query = "most hyped meme coin"
                break
        if not query:
            query = txt  # Non-crypto or general query

        # Resolve token and fetch data
        token, address, dexscreener_data = resolve_token(query)
        analysis = analyze_query(query, token, address, dexscreener_data, tid)
        reply = analysis["reply"][:200]

        # Post reply
        reply_tid = in_reply_to_status_id if in_reply_to_status_id and in_reply_to_status_id.isdigit() else tid
        try:
            tweet_response = x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid))
            logger.info(f"Replied to {reply_tid}: {reply}")
        except tweepy.errors.Forbidden as e:
            logger.warning(f"Threaded reply failed for {reply_tid}: {str(e)}; posting standalone")
            tweet_response = x_client.create_tweet(text=reply)
            logger.info(f"Standalone reply: {reply}")
        except tweepy.TweepyException as e:
            logger.error(f"Failed to post reply: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

        return JSONResponse({"message": "Replied to mention"}, status_code=200)
    except Exception as e:
        logger.error(f"handle_mention error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")