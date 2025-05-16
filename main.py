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
    "X_BEARER_TOKEN",
    "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Tweepy client with OAuth 2.0 Bearer Token
x_client = tweepy.Client(bearer_token=os.getenv("X_BEARER_TOKEN"))

# Get @askdegen's user ID
askdegen_user = x_client.get_me(user_auth=False).data
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

# Search X for token ticker to resolve contract address
def search_x_for_token(ticker: str) -> tuple:
    query = f"{ticker} solana contract address -in:replies"
    try:
        tweets = x_client.search_recent_tweets(
            query=query,
            tweet_fields=["text"],
            max_results=10,
            user_auth=False
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
                if data and data.get("token_symbol", "").upper() == ticker.replace("$", ""):
                    return ticker.replace("$", ""), address
        return None, None
    except tweepy.TweepyException as e:
        logger.error(f"X search category_id=x_search_error, error={str(e)}")
        return None, None

# Resolve token ticker or address
def resolve_token(query: str) -> tuple:
    query = query.strip().upper()
    if query in ["DEGEN", "$DEGEN"]:
        return "DEGEN", DEGEN_ADDRESS
    if address_regex.match(query):
        data = fetch_dexscreener_data(query)
        if data:
            return data["token_symbol"].upper(), query
        return None, query
    # Extract ticker (e.g., $MOODENG)
    ticker = query if query.startswith("$") else f"${query}"
    return search_x_for_token(ticker)

# Fetch real-time X sentiment and trends
def fetch_x_sentiment_and_trends(token: str) -> dict:
    query = f"${token} -in:replies lang:en"
    try:
        tweets = x_client.search_recent_tweets(
            query=query,
            tweet_fields=["text", "created_at"],
            max_results=50,
            user_auth=False
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

# Fetch current events and context from X
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
                user_auth=False
            )
            if tweets.data:
                key = ["trends", "news", "weather_impact"][i]
                context[key] = [{"text": t.text, "created_at": str(t.created_at)} for t in tweets.data]
        except tweepy.TweepyException as e:
            logger.error(f"X context category_id=x_context_error, query={query}, error={str(e)}")
    return context

# Handle a mention event
async def handle_mention(data: dict):
    try:
        evt = data["tweet_create_events"][0]
        txt = evt.get("text", "").replace("@askdegen", "").strip()
        user = evt.get("user", {}).get("screen_name", "unknown")
        tid = evt.get("id_str", "")
        reply_tid = evt.get("in_reply_to_status_id_str", tid) or tid

        if not all([txt, user, tid]):
            logger.warning(f"Invalid tweet data: {txt=}, {user=}, {tid=}")
            return {"message": "Invalid tweet"}, 400

        logger.info(f"Processing mention: {tid}, {user}, {txt}")

        post_count = int(redis_client.get(f"{REDIS_CACHE_PREFIX}post_count") or 0)
        if post_count >= 100:
            logger.warning("Daily post limit reached")
            return {"message": "Post limit reached"}, 429

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

        # Case 2: $TOKEN or contract address
        else:
            # Extract token or address
            token_query = None
            for word in txt.split():
                if word.startswith("$") or address_regex.match(word):
                    token_query = word
                    break
            if not token_query:
                token_query = "most hyped solana meme coin"

            token, address = resolve_token(token_query)
            if not address:
                reply = "Couldn't resolve token address. Try a $TICKER or contract address."
            else:
                # Fetch DexScreener data
                market_data = fetch_dexscreener_data(address)
                if not market_data:
                    reply = f"Couldn't fetch data for {token or token_query} from DexScreener."
                else:
                    # Fetch X sentiment and trends
                    x_sentiment = fetch_x_sentiment_and_trends(token or token_query.replace("$", ""))
                    # Fetch current context
                    current_context = fetch_current_context()
                    system = (
                        "Expert crypto analyst specializing in Solana meme coins. Craft a concise (≤280 chars) tweet "
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

        # Post reply
        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid), user_auth=False)
            redis_client.incr(f"{REDIS_CACHE_PREFIX}post_count")
            logger.info(f"Replied to mention: {reply} to tweet {reply_tid}")
        except tweepy.errors.Forbidden:
            x_client.create_tweet(text=reply, user_auth=False)
            redis_client.incr(f"{REDIS_CACHE_PREFIX}post_count")
            logger.info(f"Created new tweet for mention: {reply} (couldn't reply)")

        return {"message": "Success"}, 200
    except Exception as e:
        logger.error(f"Mention category_id=mention_error, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
            max_results=10,
            user_auth=False
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
                last_id = max(last_id or 0, tweet.id)
                redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", last_id)
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

async def poll_mentions_loop():
    while True:
        last_mention = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        sleep_time = 90 if last_mention and (int(time.time()) - int(last_mention) < 3600) else 1800
        await poll_mentions()
        await asyncio.sleep(sleep_time)

async def sleep_until_next_reset():
    now = time.time()
    target_time = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
    if now >= target_time:
        target_time += 86400
    sleep_duration = target_time - now
    logger.info(f"Sleeping for {sleep_duration} seconds until next reset at 9:00 AM EDT")
    await asyncio.sleep(sleep_duration)
    redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}post_count", 0)
    redis_client.set(f"{REDIS_CACHE_PREFIX}last_reset", int(time.time()))

@app.on_event("startup")
async def start_polling():
    logger.info("Starting polling for mentions...")
    await reset_daily_counters()
    asyncio.create_task(poll_mentions_loop())

async def reset_daily_counters():
    last_reset = redis_client.get(f"{REDIS_CACHE_PREFIX}last_reset")
    now = time.time()
    target_time = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
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
        return await handle_mention(test_event)
    except Exception as e:
        logger.error(f"Test category_id=test_error, error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))