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

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load env ---
load_dotenv()
for v in [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# --- Twitter + Redis setup ---
API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

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

# --- Rate limiting ---
RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
mentions_timestamps = deque()
tweet_timestamps = deque()

# --- Data Fetcher ---
def fetch_data(addr):
    try:
        url = DEXS_URL + addr
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        pair = data['pairs'][0]
        return {
            'price_usd': float(pair['priceUsd']),
            'market_cap': float(pair.get('fdv', 0)),
            'volume_usd': float(pair['volume']['h24']),
            'change_1h': float(pair['priceChange']['h1']),
            'change_24h': float(pair['priceChange']['h24']),
            'liquidity_usd': float(pair['liquidity']['usd'])
        }
    except Exception as e:
        logger.warning(f"fetch_data error: {e}")
        return {
            'price_usd': 0,
            'market_cap': 0,
            'volume_usd': 0,
            'change_1h': 0,
            'change_24h': 0,
            'liquidity_usd': 0
        }

# --- Pro/Edgy Metrics ---
def format_metrics(d):
    return (
        f"🔮 $DEGEN Insights\n"
        f"Price: ${d['price_usd']:,.4f} | MCap: ${d['market_cap']:,.0f}\n"
        f"24h Vol: ${d['volume_usd']:,.0f} | 1h: {d['change_1h']:+.2f}%\n"
        f"Pattern Recognition: {identify_chart_pattern(d)}"
    )

def identify_chart_pattern(data):
    changes = [data['change_1h'], data['change_24h']]
    if all(c > 0 for c in changes):
        return "Bullish Ascending Triangle"
    elif changes[0] > 0 > changes[1]:
        return "Bull Flag Formation"
    else:
        return "Consolidation Phase"

# --- Perplexity AI ---
def ask_perplexity(prompt):
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "sonar-medium-chat",
        "messages": [
            {
                "role": "system",
                "content": """You're a crypto analyst with deep technical insight and market savvy. Respond with:
- Chain analysis (volume, liquidity pools)
- Sentiment interpretation (social, whale activity)
- Technical patterns (chart formations, indicators)
- Concise, professional, a little edgy (no emojis)
- Never mention a knowledge base or context explicitly.
"""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.65,
        "frequency_penalty": 1.1,
        "presence_penalty": 0.9
    }
    try:
        r = requests.post("https://api.perplexity.ai/chat/completions", json=body, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")
        return "Analyzing market patterns... Check back soon. [Degen Out]"

# --- Recent Replies Memory ---
RECENT_REPLIES_KEY = f"{REDIS_PREFIX}recent_replies"
RECENT_REPLIES_LIMIT = 50

def save_recent_reply(user_text, bot_text):
    entry = json.dumps({"user": user_text, "bot": bot_text})
    db.lpush(RECENT_REPLIES_KEY, entry)
    db.ltrim(RECENT_REPLIES_KEY, 0, RECENT_REPLIES_LIMIT - 1)

def get_recent_replies(n=5):
    items = db.lrange(RECENT_REPLIES_KEY, 0, n-1)
    replies = [json.loads(x) for x in items]
    return replies

# --- Conversation Memory (per thread) ---
def get_thread_key(convo_id):
    return f"{REDIS_PREFIX}thread:{convo_id}"

def get_convo_count(convo_id):
    return int(db.hget(get_thread_key(convo_id), "count") or 0)

def increment_convo_count(convo_id):
    db.hincrby(get_thread_key(convo_id), "count", 1)
    db.expire(get_thread_key(convo_id), 86400)

def get_thread_history(convo_id):
    return db.hget(get_thread_key(convo_id), "history") or ""

def update_thread_history(convo_id, user_text, bot_text):
    history = get_thread_history(convo_id)
    new_history = (history + f"\nUser: {user_text}\nBot: {bot_text}")[-1000:]  # last 1000 chars
    db.hset(get_thread_key(convo_id), "history", new_history)
    db.expire(get_thread_key(convo_id), 86400)

# --- Rate Guard ---
async def safe_mention_lookup(fn, *args, **kwargs):
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        wait = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Mentions limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        res = fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] get_users_mentions 429; backing off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_mention_lookup(fn, *args, **kwargs)
    mentions_timestamps.append(time.time())
    return res

async def safe_tweet(text: str, **kwargs):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        wait = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Tweet limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] create_tweet 429; backing off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_tweet(text=text, **kwargs)
    tweet_timestamps.append(time.time())
    return resp

# --- Core Handlers ---
async def handle_mention(tw):
    convo_id = tw.conversation_id or tw.id
    convo_count = get_convo_count(convo_id)
    if convo_count >= 2:
        return

    history = get_thread_history(convo_id)
    recent = get_recent_replies(5)
    recent_str = "\n".join([f"User: {r['user']}\nBot: {r['bot']}" for r in recent])

    prompt = (
        f"Here are some of your most recent crypto takes:\n{recent_str}\n"
        f"{history}\nUser: {tw.text}\nBot:"
    )
    response = ask_perplexity(prompt)
    if convo_count == 1:
        response += " [Degen Out]"

    await safe_tweet(text=response[:240], in_reply_to_tweet_id=tw.id)
    update_thread_history(convo_id, tw.text, response)
    increment_convo_count(convo_id)
    save_recent_reply(tw.text, response)
    db.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

async def mention_loop():
    while True:
        try:
            last_id = db.get(f"{REDIS_PREFIX}last_mention_id")
            res = await safe_mention_lookup(
                x_client.get_users_mentions,
                id=BOT_ID,
                since_id=last_id,
                tweet_fields=['id', 'text', 'conversation_id'],
                expansions=['author_id'],
                user_fields=['username'],
                max_results=10
            )
            if res and res.data:
                for tw in reversed(res.data):
                    if not db.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        db.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                        await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            prompt = f"Create an insightful, one-line, pro/edgy comment about $DEGEN using: {json.dumps(data)}"
            insight = ask_perplexity(prompt)
            metrics = format_metrics(data)
            final = f"{metrics}\n\n💡 {insight[:140]}"
            if final != db.get(f"{REDIS_PREFIX}last_hourly_post"):
                await safe_tweet(text=final[:560])
                db.setex(f"{REDIS_PREFIX}last_hourly_post", 3600, final)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        mention_loop(),
        hourly_post_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())
