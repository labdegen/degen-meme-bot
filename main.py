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
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load env vars and validate
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY", "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# Initialize Redis
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Initialize Tweepy client (v2) & API (v1.1)
API_KEY        = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN   = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN   = os.getenv("X_BEARER_TOKEN")

x_client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_KEY_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)
me = x_client.get_me().data
BOT_ID = me.id
logger.info(f"Authenticated as: {me.username} (ID: {BOT_ID})")

oauth = tweepy.OAuth1UserHandler(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
x_api = tweepy.API(oauth)

# Rate-limit guard for ALL tweet and lookup calls
RATE_WINDOW   = 900      # 15 minutes in seconds
MENTIONS_MAX  = 10       # per-user mentions lookup cap
TWEETS_MAX    = 100      # per-user tweet cap (just to be safe)

# Deques to track timestamps of our requests
mentions_timestamps = deque()
tweet_timestamps    = deque()

async def safe_mention_lookup(fn, *args, **kwargs):
    """
    Wrap any call to get_users_mentions to enforce
    <= MENTIONS_MAX calls in the past RATE_WINDOW.
    """
    now = time.time()
    # purge old entries
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()

    if len(mentions_timestamps) >= MENTIONS_MAX:
        reset_in = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        logger.warning(f"Mentions rate limit hit. Sleeping {reset_in:.0f}s…")
        await asyncio.sleep(reset_in)

    response = fn(*args, **kwargs)
    mentions_timestamps.append(time.time())
    return response

async def safe_tweet(text: str, **kwargs):
    """
    Wrap create_tweet so we never exceed TWEETS_MAX
    in the past RATE_WINDOW—and also back off if
    Twitter returns a 429 with reset info.
    """
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()

    if len(tweet_timestamps) >= TWEETS_MAX:
        reset_in = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"Tweet-posting rate limit guard. Sleeping {reset_in:.0f}s…")
        await asyncio.sleep(reset_in)

    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset_ts = int(e.response.headers.get("x-rate-limit-reset", time.time() + RATE_WINDOW))
        wait = max(0, reset_ts - time.time()) + 1
        logger.error(f"Twitter 429 received. Backing off {wait:.0f}s…")
        await asyncio.sleep(wait)
        return await safe_tweet(text, **kwargs)

    tweet_timestamps.append(time.time())
    return resp

# Other helper funcs (ask_grok, ask_perplexity, fetch_data, resolve_token, etc.)
# … [unchanged from your original code] …

@app.get("/")
async def root():
    return {"status": "Degen bot is live."}

@app.on_event("startup")
async def startup_event():
    # Delay first poll so we don't blast at t=0
    await asyncio.sleep(60)
    asyncio.create_task(poll_loop())
    # Delay first promo so we don't tweet on startup
    asyncio.create_task(hourly_post_loop())

async def poll_loop():
    while True:
        last = db.get("degen:last_tweet_id")
        since_id = int(last) if last else None

        # Use our safe wrapper around the mentions lookup
        res = await safe_mention_lookup(
            x_client.get_users_mentions,
            id=BOT_ID,
            since_id=since_id,
            tweet_fields=['id', 'text', 'author_id'],
            expansions=['author_id'],
            user_fields=['username'],
            max_results=10
        )

        if res and res.data:
            users = {u.id: u.username for u in res.includes.get('users', [])}
            for tw in reversed(res.data):
                ev = {'tweet_create_events': [{
                    'id_str': str(tw.id),
                    'text': tw.text,
                    'user': {'screen_name': users.get(tw.author_id, '?')}
                }]}
                try:
                    await handle_mention(ev)
                except Exception as e:
                    logger.error(f"Mention error: {e}")
                db.set("degen:last_tweet_id", tw.id)
                db.set("degen:last_mention", int(time.time()))

        # Poll no faster than once every 90s
        await asyncio.sleep(90)

async def hourly_post_loop():
    # First wait one hour after startup
    await asyncio.sleep(3600)
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            card = format_metrics(d)
            context = ask_grok(
                "You're a Degen community member summarizing recent metrics. Make it casual, grounded, and complete within 2 sentences.",
                json.dumps(d),
                max_tokens=160
            )
            tweet = f"{card}\n{context}"
            if len(tweet) > 380:
                tweet = tweet[:380].rsplit('.', 1)[0] + '.'

            # Use our safe_tweet wrapper
            await safe_tweet(text=tweet)
            logger.info("Hourly promo posted")

        except Exception as e:
            logger.error(f"Promo loop error: {e}")

        await asyncio.sleep(3600)

async def handle_mention(ev: dict):
    # … [your existing logic to build `reply`] …
    # at the end, replace x_client.create_tweet with safe_tweet:
    await safe_tweet(text=reply, in_reply_to_tweet_id=int(tid))
    return {'message': 'ok'}
