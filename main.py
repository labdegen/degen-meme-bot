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
    "X_API_KEY", "X_API_KEY_SECRET",           # OAuth 1.0a Consumer Keys
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", # OAuth 1.0a Access Tokens
    "X_BEARER_TOKEN",                          # OAuth 2.0 Bearer Token (for v2 reads)
    "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Set up X client with both OAuth 1.0a and OAuth 2.0 Bearer Token
api_key              = os.getenv("X_API_KEY")
api_key_secret       = os.getenv("X_API_KEY_SECRET")
access_token         = os.getenv("X_ACCESS_TOKEN")
access_token_secret  = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token         = os.getenv("X_BEARER_TOKEN")

x_client = tweepy.Client(
    bearer_token        = bearer_token,
    consumer_key        = api_key,
    consumer_secret     = api_key_secret,
    access_token        = access_token,
    access_token_secret = access_token_secret,
)

# Create Tweepy API v1.1 object for backward compatibility if needed
x_api = tweepy.API(
    tweepy.OAuth1UserHandler(
        consumer_key        = api_key,
        consumer_secret     = api_key_secret,
        access_token        = access_token,
        access_token_secret = access_token_secret
    )
)

# Get @askdegen's user ID
try:
    askdegen_user = x_client.get_me().data
    ASKDEGEN_ID   = askdegen_user.id
    logger.info(f"Authenticated as: {askdegen_user.username}, ID: {ASKDEGEN_ID}")
except Exception as e:
    logger.error(f"Authentication failed: {str(e)}")
    raise

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

def fetch_dexscreener_data(address: str, retries=3, backoff=2) -> dict:
    """Fetch token metrics from DexScreener with caching."""
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis get category_id=redis_error, error={str(e)}")

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
                time.sleep(backoff)
                backoff *= 2
                continue
            return {}
        except requests.RequestException as e:
            logger.error(f"DexScreener category_id=dexscreener_error, error={str(e)}")
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
        token   = query.replace("$", "").upper()
        system  = "Crypto analyst. Find Solana address for $TOKEN from X. JSON: {'token': str, 'address': str}"
        user_msg = f"Ticker: {token}. Find address from X."

    cache_key = f"{REDIS_CACHE_PREFIX}resolve:{query}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data    = json.loads(cached)
            token   = data.get("token", "Unknown").upper()
            address = data.get("address", "")
            return token, address, fetch_dexscreener_data(address) if address else {}
    except redis.RedisError:
        pass

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    try:
        r    = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        token   = data.get("token", "Unknown").upper()
        address = data.get("address", "")
        redis_client.setex(cache_key, 3600, json.dumps({"token": token, "address": address}))
        return token, address, fetch_dexscreener_data(address) if address else {}
    except Exception as e:
        logger.error(f"Resolve category_id=resolve_error, error={str(e)}")
        return "Unknown", "", {}

def handle_confession(confession: str, user: str, tid: str) -> str:
    """Parse and tweet a Degen Confession."""
    system   = "Witty crypto bot. Summarize confession into a fun, anonymized tweet with a challenge. ≤750 chars, use only what's needed. JSON: {'tweet': str}"
    user_msg = f"Confession: {confession}. Hype degen spirit, add challenge, keep it short."
    headers  = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body     = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 750,
        "temperature": 0.7
    }
    try:
        r       = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data    = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        tweet   = data.get("tweet", "Degen spilled a wild tale! Share yours! #DegenConfession")[:750]
        response = x_client.create_tweet(text=tweet)
        link    = f"https://x.com/askdegen/status/{response.data['id']}"
        return f"Your confession's live! See: {link}"
    except Exception as e:
        logger.error(f"Confession category_id=confession_error, error={str(e)}")
        return "Confession failed. Try again!"

def analyze_hype(query: str, token: str, address: str, dexscreener_data: dict, tid: str) -> str:
    """Analyze hype for a coin with conversation memory."""
    context_key = f"{REDIS_CACHE_PREFIX}context:{tid}"
    try:
        context       = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError:
        prior_context = {"query": "", "response": ""}

    is_degen = token == "DEGEN" or address == DEGEN_ADDRESS
    system = (
        "Witty crypto analyst. Analyze coin hype from X and market data. For $DEGEN, stay positive, compare to $DOGE/$SHIB's ups/downs. "
        "Reply ≤150 chars, 1-2 sentences. JSON: {'reply': str, 'hype_score': int}"
    )
    user_msg = (
        f"Coin: {token}. Market: {json.dumps(dexscreener_data)}. Prior: Query: {prior_context['query']}, "
        f"Reply: {prior_context['response']}. Fun, short reply, hype score. "
        + ("Stay positive for $DEGEN." if is_degen else "")
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body    = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    try:
        r     = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        data  = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        reply = data.get("reply", "No vibe on X. Try $BONK!")[:150]
        redis_client.setex(context_key, 86400, json.dumps({"query": query, "response": reply}))
        return reply
    except Exception as e:
        logger.error(f"Hype category_id=hype_error, error={str(e)}")
        return "No vibe on X. Try $BONK!"

async def sleep_until_next_reset():
    """Sleep until the next daily reset at 9:00 AM EDT."""
    now         = time.time()
    target_time = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
    if now >= target_time:
        target_time += 86400
    sleep_duration = target_time - now
    logger.info(f"Sleeping for {sleep_duration} seconds until next reset at 9:00 AM EDT")
    await asyncio.sleep(sleep_duration)
    redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}post_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}last_reset", int(time.time()))

async def poll_mentions():
    """Poll for @askdegen mentions using GET /2/users/:id/mentions."""
    last_tweet_id = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    last_tweet_id = int(last_tweet_id) if last_tweet_id else None

    try:
        read_count = int(redis_client.get(f"{REDIS_CACHE_PREFIX}read_count") or 0)
        if read_count >= 500:
            logger.info("Daily read limit reached, waiting until 9:00 AM EDT")
            await sleep_until_next_reset()
            return
        redis_client.incrby(f"{REDIS_CACHE_PREFIX}read_count", 10)

        tweets = x_client.get_users_mentions(
            id           = ASKDEGEN_ID,
            since_id     = last_tweet_id,
            tweet_fields = ["id", "text", "author_id", "in_reply_to_status_id"],
            user_fields  = ["username"],
            expansions   = ["author_id"],
            max_results  = 10
        )

        if tweets.data:
            users = {u.id: u.username for u in tweets.includes.get("users", [])}
            for tweet in reversed(tweets.data):
                logger.info(f"Found mention: {tweet.id}, {tweet.text}")
                event = {
                    "tweet_create_events": [
                        {
                            "id_str": str(tweet.id),
                            "text": tweet.text,
                            "user": {"screen_name": users.get(tweet.author_id, "unknown")},
                            "in_reply_to_status_id_str": str(tweet.in_reply_to_status_id) if tweet.in_reply_to_status_id else None
                        }
                    ]
                }
                await handle_mention(event)
                last_tweet_id = max(last_tweet_id or 0, tweet.id)
                redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", last_tweet_id)
                redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention", int(time.time()))
        else:
            logger.info("No new mentions")

    except tweepy.TweepyException as e:
        logger.error(f"Polling mentions category_id=polling_error, error={str(e)}")
        if "Rate limit" in str(e):
            logger.info("Hit rate limit, waiting 15 minutes")
            await asyncio.sleep(900)
        else:
            await asyncio.sleep(60)

async def handle_mention(data: dict):
    """Handle @askdegen mentions and comments."""
    try:
        if "tweet_create_events" not in data or not data["tweet_create_events"]:
            logger.warning("No tweet_create_events in payload")
            return {"message": "No tweet events"}, 400

        evt       = data["tweet_create_events"][0]
        txt       = evt.get("text", "").replace("@askdegen", "").strip()
        user      = evt.get("user", {}).get("screen_name", "")
        tid       = evt.get("id_str", "")
        reply_tid = evt.get("in_reply_to_status_id_str", tid) or tid

        if not all([txt, user, tid]):
            logger.warning(f"Invalid tweet data: txt={txt}, user={user}, tid={tid}")
            return {"message": "Invalid tweet"}, 400

        logger.info(f"Processing mention: {tid}, {user}, {txt}")

        post_count = int(redis_client.get(f"{REDIS_CACHE_PREFIX}post_count") or 0)
        if post_count >= 100:
            logger.warning("Daily post limit reached")
            return {"message": "Post limit reached"}, 429

        if txt.lower().startswith("degen confession:"):
            reply = handle_confession(txt[16:].strip(), user, tid)
        else:
            query = (
                next((w[1:] for w in txt.split() if w.startswith("$") and len(w) > 1), None)
                or next((w for w in txt.split() if re.match(r"^[A-Za-z0-9]{43,44}$", w)), None)
                or "most hyped coin"
            )
            token, address, data = resolve_token(query)
            reply = analyze_hype(query, token, address, data, tid)

        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid))
            redis_client.incr(f"{REDIS_CACHE_PREFIX}post_count")
            logger.info(f"Replied to mention: {reply} to tweet {reply_tid}")
        except tweepy.errors.Forbidden:
            x_client.create_tweet(text=reply)
            redis_client.incr(f"{REDIS_CACHE_PREFIX}post_count")
            logger.info(f"Created new tweet for mention: {reply} (couldn't reply)")

        return {"message": "Success"}, 200

    except Exception as e:
        logger.error(f"Mention category_id=mention_error, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def poll_mentions_loop():
    """Loop to poll mentions with dynamic frequency."""
    while True:
        last_mention = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        sleep_time   = 90 if last_mention and (int(time.time()) - int(last_mention) < 3600) else 1800
        await poll_mentions()
        await asyncio.sleep(sleep_time)

@app.on_event("startup")
async def start_polling():
    """Start polling for mentions on app startup."""
    logger.info("Starting polling for mentions...")
    await reset_daily_counters()
    asyncio.create_task(poll_mentions_loop())

async def reset_daily_counters():
    """Reset daily counters at startup if needed."""
    last_reset = redis_client.get(f"{REDIS_CACHE_PREFIX}last_reset")
    now        = time.time()
    target_time = (
        time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S"))
        - (4 * 3600)
    )
    if last_reset and int(last_reset) >= target_time:
        return
    redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}post_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}last_reset", int(time.time()))
    logger.info("Daily counters reset")

@app.get("/")
async def root():
    return {"message": "Degen Meme Bot is live. Mention @askdegen with a $TOKEN or contract address!"}

@app.post("/test")
async def test_bot(request: Request):
    """Test the bot with a simulated mention."""
    try:
        body = await request.json()
        text = body.get("text", "@askdegen Tell me about $DEGEN")
        user = body.get("user", "test_user")
        test_event = {
            "tweet_create_events": [
                {
                    "id_str": "123456789",
                    "text": text,
                    "user": {"screen_name": user}
                }
            ]
        }
        response = await handle_mention(test_event)
        return response
    except Exception as e:
        logger.error(f"Test category_id=test_error, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
