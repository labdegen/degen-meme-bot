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
from random import choice
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Credentials
API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

# Redis client
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Twitter clients
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
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE = re.compile(r'^[A-Za-z0-9]{43,44}$')
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?search={}"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

# Rate limit windows
RATE_WINDOW = 900  # 15 minutes
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
mentions_timestamps = deque()
tweet_timestamps = deque()

# === Enhanced Memory System ===
def store_conversation(tweet, response):
    embedding = embedder.encode(f"{tweet.text} {response}")
    key = f"{REDIS_PREFIX}conv:{tweet.conversation_id or tweet.id}"
    pipeline = db.pipeline()
    pipeline.hset(key, mapping={
        "tweet_id": str(tweet.id),
        "text": f"User: {tweet.text}\nBot: {response}",
        "embedding": np.array(embedding).tobytes()
    })
    pipeline.expire(key, 604800)  # 1 week retention
    pipeline.execute()

def get_conversation_context(conversation_id, current_text):
    try:
        embedding = embedder.encode(current_text)
        query = np.array(embedding).tobytes()
        res = db.ft("conv_idx").search(
            f"@tweet_id:{conversation_id}",
            vector_query=f"embedding:[KNN 3 $vec]=>{{$yield_distance_as: score}}",
            query_params={"vec": query}
        )
        return "\n".join([doc.text for doc in res.docs])
    except:
        return ""

# === Rate Guard Helpers ===
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

# === Enhanced AI Helpers ===
def ask_perplexity(prompt, context=""):
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "sonar-medium-chat",
        "messages": [
            {
                "role": "system",
                "content": f"""You're a crypto analyst combining deep technical insight with market savvy. Respond with:
- Chain analysis (volume, liquidity pools)
- Sentiment interpretation (social, whale activity)
- Technical patterns (chart formations, indicators)
- Concise professional tone (no emojis)
- Creative analogies for complex concepts

Context:{context}"""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.65,
        "frequency_penalty": 1.1,
        "presence_penalty": 0.9
    }
    try:
        r = requests.post(PERPLEXITY_URL, json=body, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")
        return "Analyzing market patterns... Check back soon. [Degen Out]"

# === Conversation Management ===
def track_convo(tweet_id, reply_count):
    db.setex(f"{REDIS_PREFIX}convo:{tweet_id}", 86400, reply_count)

def get_convo_count(tweet_id):
    return int(db.get(f"{REDIS_PREFIX}convo:{tweet_id}") or 0)

# === Enhanced Metrics Formatting ===
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

# === Core Handlers ===
async def handle_mention(tw):
    convo_id = tw.conversation_id or tw.id
    convo_count = get_convo_count(convo_id)
    
    if convo_count >= 2:
        return
    
    context = get_conversation_context(convo_id, tw.text)
    prompt = f"Query: {tw.text}\nRespond with professional crypto analysis using: {context}"
    
    response = ask_perplexity(prompt, context)
    
    if convo_count == 1:
        response += " [Degen Out]"
    
    await safe_tweet(text=response[:240], in_reply_to_tweet_id=tw.id)
    store_conversation(tw, response)
    track_convo(convo_id, convo_count + 1)
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
            prompt = f"Create insightful one-liner about $DEGEN using: {json.dumps(data)}"
            insight = ask_perplexity(prompt)
            metrics = format_metrics(data)
            final = f"{metrics}\n\n💡 {insight[:140]}"
            
            if final != db.get(f"{REDIS_PREFIX}last_hourly_post"):
                await safe_tweet(text=final[:560])
                db.setex(f"{REDIS_PREFIX}last_hourly_post", 3600, final)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def memory_maintenance():
    while True:
        db.zremrangebyscore(f"{REDIS_PREFIX}conv_zset", "-inf", time.time()-604800)
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        mention_loop(),
        hourly_post_loop(),
        memory_maintenance()
    )

if __name__ == "__main__":
    # Initialize vector index
    try:
        schema = (
            TextField("tweet_id"),
            VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
            TextField("text")
        )
        db.ft("conv_idx").create_index(schema, definition=IndexDefinition(prefix=[f"{REDIS_PREFIX}conv:"]))
    except Exception as e:
        logger.info(f"Vector index already exists: {e}")
    
    asyncio.run(main())
