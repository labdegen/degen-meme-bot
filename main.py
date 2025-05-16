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
askdegen_user = x_client.get_me().data
ASKDEGEN_ID = askdegen_user.id
logger.info(f"Authenticated as: {askdegen_user.username}, ID: {ASKDEGEN_ID}")

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

# Regex for Solana contract addresses
address_regex = re.compile(r"^[A-Za-z0-9]{43,44}$")

# Fetch live data from DexScreener
def fetch_dexscreener_data(address: str, retries=3, backoff=2) -> dict:
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis get category_id=redis_error, error={str(e)}")

    for attempt in range(retries):
        try:
            resp = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or not data[0]:
                return {}
            pair = data[0]
            result = {
                "symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
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

# Search X for token ticker to resolve contract address (uses Bearer Token)
def search_x_for_token(ticker: str) -> tuple:
    query = f"{ticker} solana contract address -in:replies"
    try:
        tweets = x_client.search_recent_tweets(
            query=query,
            tweet_fields=["text"],
            max_results=10,
            user_auth=False  # Use Bearer Token for higher rate limits
        )
        if not tweets.data:
            return None, None
        
        for tweet in tweets.data:
            text = tweet.text
            address_match = address_regex.search(text)
            if address_match:
                address = address_match.group(0)
                # Verify with DexScreener
                data = fetch_dexscreener_data(address)
                if data and data.get("symbol", "").upper() == ticker.replace("$", ""):
                    return ticker.replace("$", ""), address
        return None, None
    except tweepy.TweepyException as e:
        logger.error(f"X search category_id=x_search_error, error={str(e)}")
        return None, None

# Resolve token symbol to address or validate address
def resolve_token(query: str) -> tuple:
    q = query.strip().upper()
    if q in ["DEGEN", "$DEGEN"]:
        return "DEGEN", DEGEN_ADDRESS
    if address_regex.match(q):
        return None, q
    # Use X search instead of Grok to resolve token
    ticker = q if q.startswith("$") else f"${q}"
    return search_x_for_token(ticker)

# Fetch real-time X sentiment and trends (uses Bearer Token)
def fetch_x_sentiment_and_trends(token: str) -> dict:
    query = f"${token} -in:replies lang:en"
    try:
        tweets = x_client.search_recent_tweets(
            query=query,
            tweet_fields=["text", "created_at"],
            max_results=50,
            user_auth=False  # Use Bearer Token
        )
        if not tweets.data:
            return {"sentiment": "neutral", "trending": False, "mentions": 0, "recent_tweets": []}

        positive_keywords = ["bullish", "moon", "pump", "buy", "up", "rocket"]
        negative_keywords = ["bearish", "dump", "sell", "down", "crash"]
        sentiment_score = 0
        recent_tweets = []
        for tweet in tweets.data:
            text = tweet.text.lower()
            recent_tweets.append({"text": tweet.text, "created_at": str(tweet.created_at)})
            for word in positive_keywords:
                if word in text:
                    sentiment_score += 1
                    break
            for word in negative_keywords:
                if word in text:
                    sentiment_score -= 1
                    break

        sentiment = "bullish" if sentiment_score > 0 else "bearish" if sentiment_score < 0 else "neutral"
        mentions = len(tweets.data)
        trending = mentions > 20  # Arbitrary threshold for "trending"

        return {
            "sentiment": sentiment,
            "trending": trending,
            "mentions": mentions,
            "recent_tweets": recent_tweets[:5]  # Top 5 recent tweets
        }
    except tweepy.TweepyException as e:
        logger.error(f"X sentiment category_id=x_sentiment_error, error={str(e)}")
        return {"sentiment": "neutral", "trending": False, "mentions": 0, "recent_tweets": []}

# Fetch current events and context from X (uses Bearer Token)
def fetch_current_context() -> dict:
    queries = [
        "solana meme coin -in:replies lang:en",
        "crypto market news -in:replies lang:en",
        "weather affecting crypto -in:replies lang:en"
    ]
    context = {"trends": [], "news": [], "weather_impact": []}
    for i, query in enumerate(queries):
        try:
            tweets = x_client.search_recent_tweets(
                query=query,
                tweet_fields=["text", "created_at"],
                max_results=10,
                user_auth=False  # Use Bearer Token
            )
            if tweets.data:
                key = ["trends", "news", "weather_impact"][i]
                context[key] = [{"text": t.text, "created_at": str(t.created_at)} for t in tweets.data]
        except tweepy.TweepyException as e:
            logger.error(f"X context category_id=x_context_error, query={query}, error={str(e)}")
    return context

# Handle a mention event
async def handle_mention(data: dict):
    evt = data["tweet_create_events"][0]
    txt = evt.get("text", "").replace("@askdegen", "").strip()
    tid = evt.get("id_str")
    user = evt.get("user", {}).get("screen_name", "unknown")
    
    # Load conversational context from Redis
    context_key = f"{REDIS_CACHE_PREFIX}context:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError:
        prior_context = {"query": "", "response": ""}

    # Case 1: Degen Confession
    if txt.lower().startswith("degen confession:"):
        system = (
            "Witty crypto bot. Summarize confession into a fun, anonymized tweet with a challenge. "
            "≤750 chars, use only what's needed. JSON: {'tweet': str}"
        )
        user_msg = f"Confession: {txt[16:].strip()}. Hype degen spirit, add challenge, keep it short."
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            "max_tokens": 750,
            "temperature": 0.7
        }
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        reply = data.get("tweet", "Degen spilled a wild tale! Share yours! #DegenConfession")[:750]
        link = f"https://x.com/askdegen/status/{tid}"
        reply = f"Your confession's live! See: {link}"
        redis_client.setex(context_key, 86400, json.dumps({"query": txt, "response": reply}))

    # Case 2: $TOKEN or on-chain address
    elif txt.startswith("$") or address_regex.search(txt):
        token, address = resolve_token(txt.split()[0])
        if not address:
            reply = "Could not resolve token address."
        else:
            market_data = fetch_dexscreener_data(address)
            if not market_data:
                reply = f"Couldn't fetch data for {token or txt.split()[0]} from DexScreener."
            else:
                # Fetch X sentiment and trends
                x_sentiment = fetch_x_sentiment_and_trends(token or txt.split()[0].replace("$", ""))
                # Fetch current context
                current_context = fetch_current_context()
                system = (
                    "Expert crypto analyst specializing in Solana meme coins with a dry degen tone. Craft a concise (≤280 chars) tweet "
                    "analyzing the token using real-time DexScreener data and X sentiment. Include specific price, volume, "
                    "and sentiment insights. Focus on unique trends from large Solana accounts on X. Avoid mentioning $DOGE, $SHIB, $PEPE. "
                    "Incorporate broader context (trends, news, weather) if relevant."
                )
                user_msg = (
                    f"Token: {token or 'Unknown'}. DexScreener: {json.dumps(market_data)}. "
                    f"X Sentiment: {json.dumps(x_sentiment)}. Current Context: {json.dumps(current_context)}. "
                    f"Prior: Query: {prior_context['query']}, Response: {prior_context['response']}. "
                    f"User Query: {txt}. Provide specific, data-driven insights."
                )
                headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
                body = {
                    "model": "grok-3",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                    "max_tokens": 280,
                    "temperature": 0.7
                }
                r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
                reply = r.json()["choices"][0]["message"]["content"].strip()
                redis_client.setex(context_key, 86400, json.dumps({"query": txt, "response": reply}))

    # Case 3: Freeform via Grok
    else:
        # Fetch current context for freeform queries
        current_context = fetch_current_context()
        system = (
            "AtlasAI Degen Bot: professional crypto assistant with dry degen humor—answer conversationally. "
            "Incorporate real-time X data on Solana meme coin trends, crypto news, and weather impacts if relevant."
        )
        user_msg = (
            f"Current Context: {json.dumps(current_context)}. "
            f"Prior: Query: {prior_context['query']}, Response: {prior_context['response']}. "
            f"User Query: {txt}."
        )
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            "max_tokens": 150,
            "temperature": 0.7
        }
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        reply = r.json()["choices"][0]["message"]["content"].strip()
        redis_client.setex(context_key, 86400, json.dumps({"query": txt, "response": reply}))

    # Tweet the reply
    try:
        x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
        logger.info(f"Replied to {tid}: {reply}")
        return {"message": "Success"}, 200
    except tweepy.errors.Forbidden as e:
        logger.error(f"Failed to reply to {tid}: {str(e)}")
        # Fallback: post as a new tweet
        x_client.create_tweet(text=reply)
        logger.info(f"Created new tweet for mention: {reply} (couldn't reply)")
        return {"message": "Success (posted as new tweet)"}, 200

# Polling loop to check mentions
async def poll_mentions():
    last_id = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    since_id = int(last_id) if last_id else None
    try:
        read_count = int(redis_client.get(f"{REDIS_CACHE_PREFIX}read_count") or 0)
        if read_count >= 500:
            logger.info("Daily read limit reached, waiting until 9:00 AM EDT")
            await sleep_until_next_reset()
            return
        redis_client.incrby(f"{REDIS_CACHE_PREFIX}read_count", 10)

        tweets = x_client.get_users_mentions(
            id=ASKDEGEN_ID,
            since_id=since_id,
            tweet_fields=["id", "text", "author_id", "in_reply_to_status_id"],
            user_fields=["username"],
            expansions=["author_id"],
            max_results=10
        )

        if not tweets or not tweets.data:
            logger.info("No new mentions")
            return
        users = {u.id: u.username for u in tweets.includes.get("users", [])}
        for tw in reversed(tweets.data):
            user = users.get(tw.author_id, "unknown")
            event = {
                "tweet_create_events": [
                    {
                        "id_str": str(tw.id),
                        "text": tw.text,
                        "user": {"screen_name": user},
                        "in_reply_to_status_id_str": str(tw.in_reply_to_status_id) if tw.in_reply_to_status_id else None
                    }
                ]
            }
            try:
                await handle_mention(event)
            except Exception as e:
                logger.error(f"Error handling mention: {e}")
            redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", tw.id)
            redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention", int(time.time()))
    except tweepy.TweepyException as e:
        logger.error(f"Polling mentions category_id=polling_error, error={str(e)}")
        if "Rate limit" in str(e):
            logger.info("Hit rate limit, waiting 15 minutes")
            await asyncio.sleep(900)
        else:
            await asyncio.sleep(60)

async def poll_mentions_loop():
    while True:
        await poll_mentions()
        lm = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        interval = 90 if (lm and time.time() - int(lm) < 3600) else 1800
        logger.info(f"Sleeping {interval}s until next poll")
        await asyncio.sleep(interval)

async def sleep_until_next_reset():
    now = time.time()
    target_time = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
    if now >= target_time:
        target_time += 86400
    sleep_duration = target_time - now
    logger.info(f"Sleeping for {sleep_duration} seconds until next reset at 9:00 AM EDT")
    await asyncio.sleep(sleep_duration)
    redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}last_reset", int(time.time()))

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