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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

API_KEY             = os.getenv("X_API_KEY")
API_KEY_SECRET      = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN        = os.getenv("X_BEARER_TOKEN")
GROK_KEY            = os.getenv("GROK_API_KEY")
PERPLEXITY_KEY      = os.getenv("PERPLEXITY_API_KEY")

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

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR   = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL     = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL     = "https://api.dexscreener.com/token-pairs/v1/solana/"

ADDR_RE      = re.compile(r'\b[A-Za-z0-9]{43,44}\b')
SYMBOL_RE    = re.compile(r'\$([A-Za-z0-9]{2,10})')

RATE_WINDOW      = 900
MENTIONS_LIMIT   = 10
TWEETS_LIMIT     = 50
mentions_timestamps = deque()
tweet_timestamps    = deque()

RECENT_REPLIES_KEY   = f"{REDIS_PREFIX}recent_replies"
RECENT_REPLIES_LIMIT = 50

# Truncate at last sentence boundary
def truncate_to_sentence(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    snippet = text[:max_length]
    for sep in ('. ', '! ', '? '):
        idx = snippet.rfind(sep)
        if idx != -1:
            return snippet[: idx + 1 ]
    return snippet

# Redis-backed recent replies/thread history
def save_recent_reply(user_text, bot_text):
    entry = json.dumps({"user": user_text, "bot": bot_text})
    db.lpush(RECENT_REPLIES_KEY, entry)
    db.ltrim(RECENT_REPLIES_KEY, 0, RECENT_REPLIES_LIMIT - 1)

def get_recent_replies(n=5):
    items = db.lrange(RECENT_REPLIES_KEY, 0, n-1)
    return [json.loads(x) for x in items]

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
    new_history = (history + f"\nUser: {user_text}\nBot: {bot_text}")[-1000:]
    db.hset(get_thread_key(convo_id), "history", new_history)
    db.expire(get_thread_key(convo_id), 86400)

# Rate-limited Tweepy wrappers
async def safe_mention_lookup(fn, *args, **kwargs):
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        wait = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        await asyncio.sleep(wait)
    try:
        res = fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_mention_lookup(fn, *args, **kwargs)
    mentions_timestamps.append(time.time())
    return res

async def safe_tweet(text: str, **kwargs):
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        await asyncio.sleep(RATE_WINDOW - (now - tweet_timestamps[0]) + 1)
    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        await asyncio.sleep(max(0, reset - time.time()) + 1)
        return await safe_tweet(text=text, **kwargs)
    tweet_timestamps.append(time.time())
    return resp

# System prompts
DEGEN_SYSTEM = (
    "You are a crypto analyst: concise, sharp, professional, genius-level. "
    "Always answer ONLY about the $DEGEN token at contract address "
    f"{DEGEN_ADDR} on Solana. Do NOT mention any other token or chain. "
    "If asked about metrics, use ONLY the data provided."
)
GENERAL_SYSTEM = (
    "You are the Stephen Hawking of crypto analysis: genius-level, smart, professional. "
    "Provide on-topic, insightful answers. When asked about a specific token or contract, "
    "include the latest DEX metrics for it."
)

def ask_with_system(system_prompt, prompt, prefer_grok=False):
    body = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {GROK_KEY if prefer_grok else PERPLEXITY_KEY}",
               "Content-Type": "application/json"}
    url = GROK_URL if prefer_grok else PERPLEXITY_URL
    try:
        r = requests.post(url, json=body, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"{'Grok' if prefer_grok else 'Perplexity'} error: {e}")
        if not prefer_grok:
            return ask_with_system(system_prompt, prompt, prefer_grok=True)
        return "Unable to provide an update at this time."

# Dex data functions
def fetch_data(addr):
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        base = data.get('baseToken', {})
        return {
            'symbol':     base.get('symbol', 'UNKNOWN'),
            'price_usd':  float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h':  float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link':       f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(d):
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%\n"
        f"{d['link']}"
    )

def lookup_address(query):
    # explicit contract?
    if ADDR_RE.fullmatch(query):
        return query
    # try Dexscreener search
    try:
        r = requests.get(DEXS_SEARCH_URL + query, timeout=10)
        r.raise_for_status()
        results = r.json()
        for tok in results.get("tokens", []):
            if tok.get("symbol", "").lower() == query.lower():
                return tok.get("contractAddress")
        if results.get("tokens"):
            return results["tokens"][0].get("contractAddress")
    except Exception:
        pass
    return None

# Core mention handler
async def handle_mention(tw):
    text = tw.text.replace("@askdegen", "").strip()
    convo_id = getattr(tw, "conversation_id", None) or tw.id

    # 1) DEGEN-specific questions
    if re.search(r"\bDEGEN\b", text, re.IGNORECASE) or re.search(r"\$DEGEN\b", text, re.IGNORECASE):
        data = fetch_data(DEGEN_ADDR)
        prompt = (
            f"Metrics: {json.dumps(data)}\n"
            f"User asked: {text}\n"
            "Using ONLY these metrics, respond under 240 characters."
        )
        raw = ask_with_system(DEGEN_SYSTEM, prompt, prefer_grok=True)
        reply = truncate_to_sentence(raw, 240)
        await safe_tweet(text=reply, in_reply_to_tweet_id=tw.id)
        return

    # 2) Any other token by symbol or contract
    addr_match = ADDR_RE.search(text)
    sym_match  = SYMBOL_RE.search(text)
    if addr_match or sym_match:
        addr = addr_match.group(0) if addr_match else lookup_address(sym_match.group(1))
        if addr:
            data = fetch_data(addr)
            await safe_tweet(text=format_metrics(data), in_reply_to_tweet_id=tw.id)
            return

    # 3) General queries
    prompt = (
        f"User: {text}\n"
        "As the Stephen Hawking of crypto analysis, provide an insightful, on-topic answer."
    )
    raw = ask_with_system(GENERAL_SYSTEM, prompt)
    reply = truncate_to_sentence(raw, 240)
    await safe_tweet(text=reply, in_reply_to_tweet_id=tw.id)

# Loops
async def mention_loop():
    while True:
        try:
            last_id = db.get(f"{REDIS_PREFIX}last_mention_id")
            res = await safe_mention_lookup(
                x_client.get_users_mentions,
                id=BOT_ID,
                since_id=last_id,
                tweet_fields=['id','text'],
                expansions=['author_id'],
                user_fields=['username'],
                max_results=10
            )
            if res and res.data:
                for tw in reversed(res.data):
                    if db.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    db.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
                    db.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            prompt = (
                f"Metrics: {json.dumps(data)}\n"
                "Write a punchy one-sentence update on $DEGEN at the contract above. "
                "End on a complete sentence."
            )
            raw = ask_with_system(DEGEN_SYSTEM, prompt, prefer_grok=True)
            tweet = truncate_to_sentence(f"{metrics}\n\n{raw}", 560)

            last = db.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet.strip() != last:
                await safe_tweet(text=tweet)
                db.set(f"{REDIS_PREFIX}last_hourly_post", tweet.strip())
                logger.info("Hourly post success")
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        mention_loop(),
        hourly_post_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
