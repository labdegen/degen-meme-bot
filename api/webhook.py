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
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()
required_vars = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Tweepy client (@askdegen, Basic $200/month plan)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)
logger.info(f"Authenticated as: {x_client.get_me().data.username}")

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

def fetch_dexscreener_data(address: str, retries=3, backoff=2) -> dict:
    """Fetch token metrics from DexScreener with caching."""
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis get error: {e}")

    for attempt in range(retries):
        try:
            r = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or not data[0]:
                return {}
            pair = data[0]
            result = {
                "token_symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
                "price_usd": float(pair.get("priceUsd", 0)),
                "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                "volume_usd": float(pair.get("volume", {}).get("h24", 0)),
                "market_cap": float(pair.get("marketCap", 0))
            }
            redis_client.setex(cache_key, 300, json.dumps(result))
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                sleep(backoff)
                backoff *= 2
                continue
            return {}
        except requests.RequestException as e:
            logger.error(f"DexScreener error: {e}")
            return {}
    return {}

def resolve_token(query: str) -> tuple:
    """Resolve query to $TOKEN and address via X sentiment."""
    query = query.strip().lower()
    is_contract = re.match(r"^[A-Za-z0-9]{43,44}$", query)
    is_degen = query in ["degen", "$degen"] or (is_contract and query == DEGEN_ADDRESS)

    if is_degen:
        data = fetch_dexscreener_data(DEGEN_ADDRESS)
        return "DEGEN", DEGEN_ADDRESS, data
    elif is_contract:
        data = fetch_dexscreener_data(query)
        if data.get("token_symbol", "Unknown") != "Unknown":
            return data["token_symbol"].upper(), query, data
        system = "Crypto analyst. Find $TOKEN ticker for Solana address from X. JSON: {'token': str, 'address': str}"
        user_msg = f"Contract: {query}. Find ticker from X."
    else:
        token = query.replace("$", "").upper()
        system = "Crypto analyst. Find Solana address for $TOKEN from X. JSON: {'token': str, 'address': str}"
        user_msg = f"Ticker: {token}. Find address from X."

    cache_key = f"{REDIS_CACHE_PREFIX}resolve:{query}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            token = data.get("token", "UNKNOWN").upper()
            address = data.get("address", "")
            return token, address, fetch_dexscreener_data(address) if address else {}
    except redis.RedisError:
        pass

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 100, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        token = data.get("token", "UNKNOWN").upper()
        address = data.get("address", "")
        redis_client.setex(cache_key, 3600, json.dumps({"token": token, "address": address}))
        return token, address, fetch_dexscreener_data(address) if address else {}
    except Exception as e:
        logger.error(f"Resolve error: {e}")
        return "UNKNOWN", "", {}

def handle_confession(confession: str, user: str, tid: str) -> str:
    """Parse and tweet a Degen Confession."""
    system = "Witty crypto bot. Summarize confession into a fun, anonymized tweet with a challenge. ≤750 chars, use only what's needed. JSON: {'tweet': str}"
    user_msg = f"Confession: {confession}. Hype degen spirit, add challenge, keep it short."
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 750, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        tweet = data.get("tweet", "Degen spilled a wild tale! Share yours! #DegenConfession")[:750]
        tweet_response = x_client.create_tweet(text=tweet)
        link = f"https://x.com/askdegen/status/{tweet_response.data['id']}"
        return f"Your confession’s live! See: {link}"
    except Exception as e:
        logger.error(f"Confession error: {e}")
        return "Confession failed. Try again!"

def analyze_hype(query: str, token: str, address: str, dexscreener_data: dict, tid: str) -> str:
    """Analyze hype for a coin with conversation memory."""
    context_key = f"{REDIS_CACHE_PREFIX}context:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError:
        prior_context = {"query": "", "response": ""}

    is_degen = token == "DEGEN" or address == DEGEN_ADDRESS
    system = (
        "Witty crypto analyst. Analyze coin hype from X and market data. For $DEGEN, stay positive, compare to $DOGE/$SHIB’s ups/downs. "
        "Reply ≤150 chars, 1-2 sentences. JSON: {'reply': str, 'hype_score': int}"
    )
    user_msg = (
        f"Coin: {token}. Market: {json.dumps(dexscreener_data)}. Prior: Query: {prior_context['query']}, Reply: {prior_context['response']}. "
        f"Fun, short reply, hype score. {'Stay positive for $DEGEN.' if is_degen else ''}"
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 150, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        reply = data.get("reply", "No vibe on X. Try $BONK!")[:150]
        redis_client.setex(context_key, 86400, json.dumps({"query": query, "response": reply}))
        return reply
    except Exception as e:
        logger.error(f"Hype error: {e}")
        return "No vibe on X. Try $BONK!"

@app.post("/")
async def handle_mention(data: dict):
    """Handle @askdegen mentions and comments."""
    try:
        evt = data.get("tweet_create_events", [{}])[0]
        txt = evt.get("text", "").replace("@askdegen", "").strip()
        user = evt.get("user", {}).get("screen_name", "")
        tid = evt.get("id_str", "")
        reply_tid = evt.get("in_reply_to_status_id_str", tid) or tid

        if not all([txt, user, tid]):
            return JSONResponse({"message": "Invalid tweet"}, status_code=400)

        logger.info(f"Processing: {tid}, {user}, {txt}")

        if txt.lower().startswith("degen confession:"):
            reply = handle_confession(txt[16:].strip(), user, tid)
        else:
            query = next((w[1:] for w in txt.split() if w.startswith("$") and len(w) > 1), None) or \
                    next((w for w in txt.split() if re.match(r"^[A-Za-z0-9]{43,44}$", w)), None) or \
                    "most hyped coin"
            token, address, data = resolve_token(query)
            reply = analyze_hype(query, token, address, data, tid)

        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid))
        except tweepy.errors.Forbidden:
            x_client.create_tweet(text=reply)
        logger.info(f"Replied: {reply}")
        return JSONResponse({"message": "Success"}, status_code=200)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))