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

REDIS_PREFIX = "degen:"
DEGEN_ADDR   = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE      = re.compile(r'^[A-Za-z0-9]{43,44}$')
GROK_URL     = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_URL     = "https://api.dexscreener.com/token-pairs/v1/solana/"

RATE_WINDOW      = 900
MENTIONS_LIMIT   = 10
TWEETS_LIMIT     = 50
mentions_timestamps = deque()
tweet_timestamps    = deque()

RECENT_REPLIES_KEY   = f"{REDIS_PREFIX}recent_replies"
RECENT_REPLIES_LIMIT = 50

def truncate_to_sentence(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    snippet = text[:max_length]
    for sep in ('. ', '! ', '? '):
        idx = snippet.rfind(sep)
        if idx != -1:
            return snippet[: idx + 1 ]
    return snippet

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

def ask_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a crypto analyst: concise, sharp, professional, and a bit edgy. "
                    "Always answer ONLY about the $DEGEN token at contract address "
                    "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f on Solana. "
                    "Do NOT mention or describe any other token or chain. "
                    "If unsure, simply state: 'I only cover $DEGEN at that address.'"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    try:
        r = requests.post(PERPLEXITY_URL, json=body, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")
        return ask_grok(prompt)

def ask_grok(prompt):
    history_key = f"{REDIS_PREFIX}grok_history"
    past = set(db.lrange(history_key, 0, -1))
    body = {
        "model": "grok-3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a crypto analyst: concise, sharp, professional, and a bit edgy. "
                    "Always answer ONLY about the $DEGEN token at contract address "
                    "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f on Solana. "
                    "Do NOT mention or describe any other token or chain. "
                    "If unsure, simply state: 'I only cover $DEGEN at that address.'"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=25)
        r.raise_for_status()
        reply = r.json()['choices'][0]['message']['content'].strip()
        if reply not in past:
            db.lpush(history_key, reply)
            db.ltrim(history_key, 0, 25)
        return reply
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return "Unable to provide an update at this time."

def fetch_data(addr=DEGEN_ADDR):
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        base = data.get('baseToken', {})
        return {
            'symbol': base.get('symbol', 'DEGEN'),
            'price_usd': float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h': float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link': f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(d):
    return (
        f"🚀 {d.get('symbol', 'DEGEN')} | ${d.get('price_usd', 0):,.6f}\n"
        f"MC ${d.get('market_cap', 0):,.0f} | Vol24 ${d.get('volume_usd', 0):,.0f}\n"
        f"1h {'🟢' if d.get('change_1h', 0) >= 0 else '🔴'}{d.get('change_1h', 0):+.2f}% | "
        f"24h {'🟢' if d.get('change_24h', 0) >= 0 else '🔴'}{d.get('change_24h', 0):+.2f}%\n"
        f"{d.get('link', '')}"
    )

async def post_raid(tweet):
    text_low = tweet.text.lower()
    if '@askdegen' in text_low and 'raid' in text_low:
        prompt = (
            "Write a short one-liner hype for $DEGEN on Solana based on this: "
            f"'{tweet.text}'. Mention @ogdegenonsol but don't say 'raid'."
        )
        msg = ask_perplexity(prompt)
        img_list = glob.glob("raid_images/*.jpg")
        media = x_api.media_upload(choice(img_list)) if img_list else None
        await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            in_reply_to_tweet_id=tweet.id,
            media_ids=[media.media_id_string] if media else None
        )
        db.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
        logger.info("Raid reply sent")

async def handle_mention(tw):
    convo_id     = getattr(tw, 'conversation_id', None) or tw.id
    convo_count  = get_convo_count(convo_id)
    if convo_count >= 2:
        return

    history = get_thread_history(convo_id)
    recent  = get_recent_replies(5)
    recent_str = "\n".join([f"User: {r['user']}\nBot: {r['bot']}" for r in recent])

    prompt = (
        f"Here are some of your most recent crypto takes:\n{recent_str}\n"
        f"{history}\nUser: {tw.text}\nBot:\n"
        "Please respond in a complete, professional sentence or two, "
        "and keep the total reply under 240 characters."
    )
    raw_reply = ask_perplexity(prompt)
    reply     = truncate_to_sentence(raw_reply, 240)

    await safe_tweet(text=reply, in_reply_to_tweet_id=tw.id)
    update_thread_history(convo_id, tw.text, reply)
    increment_convo_count(convo_id)
    save_recent_reply(tw.text, reply)
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
                    tid = tw.id
                    if db.sismember(f"{REDIS_PREFIX}replied_ids", str(tid)):
                        continue
                    db.set(f"{REDIS_PREFIX}last_mention_id", tid)
                    text_low = tw.text.lower()

                    if '@askdegen' in text_low and 'raid' in text_low:
                        await post_raid(tw)
                        continue

                    txt = tw.text.replace('@askdegen', '').strip()
                    if "$DEGEN" in txt.upper() or "DEGEN" in txt.upper():
                        if "ca" in txt.lower() or "contract" in txt.lower():
                            msg = f"Contract Address: {DEGEN_ADDR}"
                        else:
                            data = fetch_data(DEGEN_ADDR)
                            msg = ask_perplexity(
                                f"User asked: {txt}\n"
                                f"Metrics: {json.dumps(data)}\n"
                                "Please respond in a complete, professional sentence or two, "
                                "and keep the total reply under 240 characters."
                            )
                        reply = truncate_to_sentence(msg, 240)
                        await safe_tweet(text=reply, in_reply_to_tweet_id=tid)
                        db.sadd(f"{REDIS_PREFIX}replied_ids", str(tid))
                    elif txt.lower() in ('ca', 'contract', 'contract address'):
                        await safe_tweet(text=f"Contract Address: {DEGEN_ADDR}", in_reply_to_tweet_id=tid)
                        db.sadd(f"{REDIS_PREFIX}replied_ids", str(tid))
                    elif txt.upper() == 'DEX':
                        data = fetch_data(DEGEN_ADDR)
                        await safe_tweet(text=format_metrics(data), in_reply_to_tweet_id=tid)
                        db.sadd(f"{REDIS_PREFIX}replied_ids", str(tid))
                    else:
                        await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def hourly_post_loop():
    while True:
        try:
            data    = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            prompt  = (
                "Respond ONLY about $DEGEN token at contract address "
                "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f on Solana. "
                "Give a punchy one-sentence update on $DEGEN using these metrics, vary every hour. "
                "Please keep the total tweet under 560 characters and end on a complete sentence."
            )
            raw     = ask_perplexity(prompt)
            combined = f"{metrics}\n\n{raw}"
            tweet   = truncate_to_sentence(combined, 560)

            last = db.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet.strip() != last:
                await safe_tweet(text=tweet)
                db.set(f"{REDIS_PREFIX}last_hourly_post", tweet.strip())
                logger.info("Hourly post success")
            else:
                logger.info("Skipped duplicate hourly post.")
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
