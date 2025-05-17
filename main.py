from fastapi import FastAPI
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

# API endpoints
GROK_URL    = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_URL    = "https://api.dexscreener.com/token-pairs/v1/solana/"
SEARCH_URL  = "https://api.dexscreener.com/latest/dex/search?search={}"

# Credentials
API_KEY             = os.getenv("X_API_KEY")
API_KEY_SECRET      = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN        = os.getenv("X_BEARER_TOKEN")
GROK_KEY            = os.getenv("GROK_API_KEY")
PERPLEXITY_KEY      = os.getenv("PERPLEXITY_API_KEY")

# Redis client
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Initialize Tweepy
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

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR   = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE      = re.compile(r'^[A-Za-z0-9]{43,44}$')

# === RATE LIMIT GUARDS ===
RATE_WINDOW     = 15 * 60    # 900 seconds
MENTIONS_LIMIT  = 10         # calls to get_users_mentions per RATE_WINDOW
TWEETS_LIMIT    = 50         # safe cap for tweets per RATE_WINDOW

mentions_timestamps = deque()
tweet_timestamps    = deque()

async def safe_mention_lookup(fn, *args, **kwargs):
    """Ensure no more than MENTIONS_LIMIT lookups per RATE_WINDOW."""
    now = time.time()
    # purge old
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        wait = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Mentions limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    res = fn(*args, **kwargs)
    mentions_timestamps.append(time.time())
    return res

async def safe_tweet(text: str, **kwargs):
    """Ensure no more than TWEETS_LIMIT tweets per RATE_WINDOW."""
    now = time.time()
    # purge old
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        wait = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Tweet limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)

    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset_ts = int(e.response.headers.get("x-rate-limit-reset", time.time() + RATE_WINDOW))
        wait = max(0, reset_ts - time.time()) + 1
        logger.error(f"[RateGuard] 429 from Twitter; backing off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_tweet(text=text, **kwargs)

    tweet_timestamps.append(time.time())
    return resp

# === YOUR EXISTING HELPERS (unchanged) ===

def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    # … your existing implementation …

def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    # … your existing implementation …

def fetch_data(addr: str) -> dict:
    # … your existing implementation …

def resolve_token(q: str) -> tuple:
    # … your existing implementation …

def format_metrics(data: dict) -> str:
    # … your existing implementation …

def format_convo_reply(data: dict, question: str) -> str:
    # … your existing implementation …

# === FASTAPI ENDPOINTS & LOOPS ===

@app.get("/")
async def root():
    return {"status": "Degen bot is live."}

@app.on_event("startup")
async def startup_event():
    # delay initial polling & promo to avoid t=0 blasts
    await asyncio.sleep(60)
    asyncio.create_task(poll_loop())
    await asyncio.sleep(3600)
    asyncio.create_task(hourly_post_loop())

async def poll_loop():
    while True:
        last    = db.get(f"{REDIS_PREFIX}last_tweet_id")
        since_id = int(last) if last else None

        # use safe lookup
        res = await safe_mention_lookup(
            x_client.get_users_mentions,
            id=BOT_ID,
            since_id=since_id,
            tweet_fields=['id','text','author_id'],
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
                db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))

        await asyncio.sleep(90)

async def hourly_post_loop():
    while True:
        try:
            d       = fetch_data(DEGEN_ADDR)
            card    = format_metrics(d)
            context = ask_grok(
                "You're a Degen community member summarizing recent metrics. "
                "Make it casual, grounded, and complete within 2 sentences.",
                json.dumps(d),
                max_tokens=160
            )
            tweet = f"{card}\n{context}"
            if len(tweet) > 380:
                tweet = tweet[:380].rsplit('.',1)[0] + '.'

            await safe_tweet(text=tweet)
            logger.info("Hourly promo posted")

        except Exception as e:
            logger.error(f"Promo loop error: {e}")

        await asyncio.sleep(3600)

async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or not isinstance(events, list) or not events[0].get("text"):
        logger.warning("Skipping invalid or empty mention event")
        return {"message":"no valid mention"}

    txt = events[0]['text'].replace('@askdegen','').strip()
    tid = events[0]['id_str']

    # build reply
    token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    if token:
        sym, addr = resolve_token(token)
        if addr:
            d = fetch_data(addr)
            reply = format_metrics(d) if txt.strip()==token else format_convo_reply(d, txt)
        else:
            reply = ask_perplexity(
                "You are a crypto researcher. Answer this tweet in under 240 characters clearly.",
                txt, max_tokens=160
            )
    else:
        reply = ask_grok("Professional crypto professor: concise analytical response.", txt, max_tokens=160)

    # trim to 240
    tweet = reply.strip()
    if len(tweet)>240:
        tweet = tweet[:240].rsplit('.',1)[0]+'.' if '.' in tweet else tweet[:240].rsplit(' ',1)[0]+'...'

    await safe_tweet(text=tweet, in_reply_to_tweet_id=int(tid))
    return {'message':'ok'}
