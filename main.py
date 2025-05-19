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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
required = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for var in required:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Twitter API setup
oauth = tweepy.OAuth1UserHandler(
    os.getenv("X_API_KEY"),
    os.getenv("X_API_KEY_SECRET"),
    os.getenv("X_ACCESS_TOKEN"),
    os.getenv("X_ACCESS_TOKEN_SECRET")
)
x_api = tweepy.API(oauth)

x_client = tweepy.Client(
    bearer_token=os.getenv("X_BEARER_TOKEN"),
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_KEY_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)
me = x_client.get_me().data
BOT_ID = me.id
BOT_USERNAME = me.username
logger.info(f"Authenticated as: {BOT_USERNAME} (ID: {BOT_ID})")

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
redis_client.ping()
logger.info("Redis connected")

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_SEARCH_URL = "https://api.dexscreener.com/api/search?query="
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

ADDR_RE = re.compile(r"\b[A-Za-z0-9]{43,44}\b")
SYMBOL_RE = re.compile(r"\$([A-Za-z0-9]{2,10})", re.IGNORECASE)
USERNAME_RE = re.compile(rf"@{BOT_USERNAME}\b", re.IGNORECASE)  # Match bot's username

RATE_WINDOW = 900
MENTIONS_LIMIT = 10
TWEETS_LIMIT = 50
SEARCH_LIMIT = 10  # Limit for search API calls
mentions_timestamps = deque()
tweet_timestamps = deque()
search_timestamps = deque()

# Set initial search ID to current time-based ID to avoid the "since_id too old" error
# Twitter IDs are roughly time-based, so this gives us a recent starting point
current_time_ms = int(time.time() * 1000) - 1728000000  # Adjust for Twitter's epoch
INITIAL_SEARCH_ID = str((current_time_ms << 22))

# Helpers

def truncate_to_sentence(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    snippet = text[:max_length]
    for sep in ('. ', '! ', '? '):
        idx = snippet.rfind(sep)
        if idx != -1:
            return snippet[:idx+1]
    return snippet

# Thread memory helpers
def get_thread_key(cid):
    return f"{REDIS_PREFIX}thread:{cid}"

def get_thread_history(cid):
    return redis_client.hget(get_thread_key(cid), "history") or ""

def increment_thread(cid):
    redis_client.hincrby(get_thread_key(cid), "count", 1)
    redis_client.expire(get_thread_key(cid), 86400)

def update_thread(cid, user_text, bot_text):
    hist = get_thread_history(cid)
    entry = f"\nUser: {user_text}\nBot: {bot_text}"
    new_hist = (hist + entry)[-2000:]
    redis_client.hset(get_thread_key(cid), "history", new_hist)
    redis_client.expire(get_thread_key(cid), 86400)

# Grok prompt
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. "
    f"Always speak about the $DEGEN token at contract address {DEGEN_ADDR}. "
    "Do NOT mention any other token or chain."
)

def ask_grok(prompt: str) -> str:
    payload = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    headers = {"Authorization": f"Bearer {os.getenv('GROK_API_KEY')}", "Content-Type": "application/json"}
    try:
        r = requests.post(GROK_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}")
        return "Unable to provide an update at this time."

async def safe_api_call(fn, timestamps_queue, limit, *args, **kwargs):
    now = time.time()
    while timestamps_queue and now - timestamps_queue[0] > RATE_WINDOW:
        timestamps_queue.popleft()
    if len(timestamps_queue) >= limit:
        await asyncio.sleep(RATE_WINDOW - (now - timestamps_queue[0]) + 1)
    try:
        return fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        await asyncio.sleep(reset - time.time() + 1)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.BadRequest as e:
        # Pass BadRequest up to be handled by the caller
        raise e
    except Exception as e:
        logger.error(f"API call error: {e}", exc_info=True)
        raise e
    finally:
        timestamps_queue.append(time.time())

async def safe_mention_lookup(fn, *args, **kwargs):
    return await safe_api_call(fn, mentions_timestamps, MENTIONS_LIMIT, *args, **kwargs)

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, search_timestamps, SEARCH_LIMIT, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, **kwargs):
    return await safe_api_call(
        lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
        tweet_timestamps, 
        TWEETS_LIMIT,
        text, 
        media_id, 
        **kwargs
    )

# DEX helpers
def fetch_data(addr: str) -> dict:
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        data = r.json()[0] if isinstance(r.json(), list) else r.json()
        base = data.get('baseToken', {})
        return {
            'symbol': base.get('symbol','DEGEN'),
            'price_usd': float(data.get('priceUsd',0)),
            'volume_usd': float(data.get('volume',{}).get('h24',0)),
            'market_cap': float(data.get('marketCap',0)),
            'change_1h': float(data.get('priceChange',{}).get('h1',0)),
            'change_24h': float(data.get('priceChange',{}).get('h24',0)),
            'link': f"https://dexscreener.com/solana/{addr}"
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {}

def format_metrics(d: dict) -> str:
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%\n"
    )

def lookup_address(token: str) -> str:
    t = token.lstrip('$')
    if t.upper() == 'DEGEN': return DEGEN_ADDR
    if ADDR_RE.fullmatch(t): return t
    try:
        r = requests.get(DEXS_SEARCH_URL + t, timeout=10)
        r.raise_for_status()
        toks = r.json().get('tokens', [])
        for item in toks:
            if item.get('symbol','').lower() == t.lower():
                return item.get('contractAddress')
        if toks: return toks[0].get('contractAddress')
    except:
        pass
    return None

def build_dex_reply(addr: str) -> str:
    data = fetch_data(addr)
    return format_metrics(data) + data['link']

async def post_raid(tweet):
    """
    Include thread history in raid prompt and post a bullpost with a random meme.
    """
    convo_id = tweet.conversation_id or tweet.id
    history = get_thread_history(convo_id)
    prompt = (
        f"History:{history}\n"
        f"User: '{tweet.text}'\n"
        "Write a one-liner bullpost for $DEGEN based on the above. "
        f"Tag @ogdegenonsol and include contract address {DEGEN_ADDR}. End with NFA."
    )
    msg = ask_grok(prompt)
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    await safe_tweet(
        text=truncate_to_sentence(msg, 240),
        media_id=media_id,
        in_reply_to_tweet_id=tweet.id
    )
    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def handle_mention(tw):
    convo_id = tw.conversation_id or tw.id
    if redis_client.hget(get_thread_key(convo_id), "count") is None:
        try:
            root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
            update_thread(convo_id, f"ROOT: {root}", "")
        except:
            update_thread(convo_id, f"ROOT: Unknown", "")
    history = get_thread_history(convo_id)
    txt = re.sub(rf"@{BOT_USERNAME}\b", "", tw.text, flags=re.IGNORECASE).strip()

    # 1) raid
    if re.search(r"\braid\b", txt, re.IGNORECASE):
        await post_raid(tw)
        return

    # 2) Handle specific commands
    # CA command - return just the contract address
    if re.match(r"\s*ca\s*$", txt, re.IGNORECASE):
        await safe_tweet(
            text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
            in_reply_to_tweet_id=tw.id
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return
        
    # DEX command - return DEX data for $DEGEN
    if re.match(r"\s*dex\s*$", txt, re.IGNORECASE):
        await safe_tweet(
            text=build_dex_reply(DEGEN_ADDR),
            in_reply_to_tweet_id=tw.id
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return
        
    # Contract Address or other similar terms - return contract address
    if re.search(r"\b(contract|address)\b", txt, re.IGNORECASE) and not re.search(r"\bdex\b", txt, re.IGNORECASE):
        await safe_tweet(
            text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
            in_reply_to_tweet_id=tw.id
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        return

    # 3) token/address -> DEX preview
    token = next((w for w in txt.split() if w.startswith('

    # 4) general fallback
    prompt = (
        f"History:{history}\n"
        f"User asked: \"{txt}\"\n"
        "First, answer naturally and concisely. "
        "Then, in a second gambler-style line, mention stacking $DEGEN. End with NFA."
    )
    raw = ask_grok(prompt)
    
    # Ensure we have a complete response that doesn't get cut off
    reply_body = raw.strip()
    
    # Make sure the response contains $DEGEN mention and contract address
    if "$DEGEN" not in reply_body:
        reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
    else:
        # If $DEGEN is already mentioned, just add the contract address if needed
        if DEGEN_ADDR not in reply_body:
            reply = f"{reply_body}\n\nContract Address: {DEGEN_ADDR}"
        else:
            reply = reply_body
    
    # Ensure we're not exceeding Twitter's character limit
    if len(reply) > 260:
        reply = truncate_to_sentence(reply, 220) + f"\n\nContract Address: {DEGEN_ADDR}"
    
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    
    await safe_tweet(
        text=reply,
        media_id=media_id,
        in_reply_to_tweet_id=tw.id
    )
    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
    update_thread(convo_id, txt, reply)
    increment_thread(convo_id)

async def mention_loop():
    while True:
        try:
            last = redis_client.get(f"{REDIS_PREFIX}last_mention_id")
            params = {
                "id": BOT_ID,
                "tweet_fields": ["id", "text", "conversation_id"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            if last:
                params["since_id"] = int(last)
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            if res and res.data:
                for tw in reversed(res.data):
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    redis_client.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def search_mentions_loop():
    """
    New loop to handle searching for mentions that might not be captured by the mentions API,
    especially mentions in communities.
    """
    # Initialize last_search_id if not present
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
        logger.info(f"Initialized last_search_id to {INITIAL_SEARCH_ID}")
    
    while True:
        try:
            query = f"@{BOT_USERNAME} -is:retweet"
            search_params = {
                "query": query,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            
            try:
                # Always try without since_id first to avoid the age restriction error
                search_results = await safe_search(
                    x_client.search_recent_tweets,
                    **search_params
                )
                logger.info("Successfully searched for community mentions")
            except Exception as e:
                logger.error(f"Search call failed: {e}", exc_info=True)
                search_results = None
            
            if search_results and search_results.data:
                newest_id = max(int(tw.id) for tw in search_results.data) if search_results.data else 0
                
                for tw in search_results.data:
                    # Skip if we've already processed this tweet
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    # Get the full tweet text from tw
                    logger.info(f"Processing community mention: {tw.id} - {tw.text[:30]}...")
                    
                    # Process the mention
                    await handle_mention(tw)
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                # Update the last search ID
                if newest_id > 0:
                    redis_client.set(f"{REDIS_PREFIX}last_search_id", str(newest_id))
                    logger.info(f"Updated last_search_id to {newest_id}")
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        # Wait before next search
        await asyncio.sleep(180)  # Run every 3 minutes

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            raw = ask_grok("Write a one-sentence bullpost update on $DEGEN. Be promotional.")
            tweet = truncate_to_sentence(metrics + raw, 560)
            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                img = choice(glob.glob("raid_images/*.jpg"))
                media_id = x_api.media_upload(img).media_id_string
                await safe_tweet(tweet, media_id=media_id)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(mention_loop(), search_mentions_loop(), hourly_post_loop())

if __name__ == "__main__":
    asyncio.run(main())) or ADDR_RE.match(w)), None)
    if token:
        sym = token.lstrip('

    # 4) general fallback
    prompt = (
        f"History:{history}\n"
        f"User asked: \"{txt}\"\n"
        "First, answer naturally and concisely. "
        "Then, in a second gambler-style line, mention stacking $DEGEN. End with NFA."
    )
    raw = ask_grok(prompt)
    
    # Ensure we have a complete response that doesn't get cut off
    reply_body = raw.strip()
    
    # Make sure the response contains $DEGEN mention and contract address
    if "$DEGEN" not in reply_body:
        reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
    else:
        # If $DEGEN is already mentioned, just add the contract address if needed
        if DEGEN_ADDR not in reply_body:
            reply = f"{reply_body}\n\nContract Address: {DEGEN_ADDR}"
        else:
            reply = reply_body
    
    # Ensure we're not exceeding Twitter's character limit
    if len(reply) > 260:
        reply = truncate_to_sentence(reply, 220) + f"\n\nContract Address: {DEGEN_ADDR}"
    
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    
    await safe_tweet(
        text=reply,
        media_id=media_id,
        in_reply_to_tweet_id=tw.id
    )
    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
    update_thread(convo_id, txt, reply)
    increment_thread(convo_id)

async def mention_loop():
    while True:
        try:
            last = redis_client.get(f"{REDIS_PREFIX}last_mention_id")
            params = {
                "id": BOT_ID,
                "tweet_fields": ["id", "text", "conversation_id"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            if last:
                params["since_id"] = int(last)
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            if res and res.data:
                for tw in reversed(res.data):
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    redis_client.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def search_mentions_loop():
    """
    New loop to handle searching for mentions that might not be captured by the mentions API,
    especially mentions in communities.
    """
    # Initialize last_search_id if not present
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
        logger.info(f"Initialized last_search_id to {INITIAL_SEARCH_ID}")
    
    while True:
        try:
            query = f"@{BOT_USERNAME} -is:retweet"
            search_params = {
                "query": query,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            
            try:
                # Always try without since_id first to avoid the age restriction error
                search_results = await safe_search(
                    x_client.search_recent_tweets,
                    **search_params
                )
                logger.info("Successfully searched for community mentions")
            except Exception as e:
                logger.error(f"Search call failed: {e}", exc_info=True)
                search_results = None
            
            if search_results and search_results.data:
                newest_id = max(int(tw.id) for tw in search_results.data) if search_results.data else 0
                
                for tw in search_results.data:
                    # Skip if we've already processed this tweet
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    # Get the full tweet text from tw
                    logger.info(f"Processing community mention: {tw.id} - {tw.text[:30]}...")
                    
                    # Process the mention
                    await handle_mention(tw)
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                # Update the last search ID
                if newest_id > 0:
                    redis_client.set(f"{REDIS_PREFIX}last_search_id", str(newest_id))
                    logger.info(f"Updated last_search_id to {newest_id}")
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        # Wait before next search
        await asyncio.sleep(180)  # Run every 3 minutes

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            raw = ask_grok("Write a one-sentence bullpost update on $DEGEN. Be promotional.")
            tweet = truncate_to_sentence(metrics + raw, 560)
            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                img = choice(glob.glob("raid_images/*.jpg"))
                media_id = x_api.media_upload(img).media_id_string
                await safe_tweet(tweet, media_id=media_id)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(mention_loop(), search_mentions_loop(), hourly_post_loop())

if __name__ == "__main__":
    asyncio.run(main())).upper()
        addr = DEGEN_ADDR if sym=="DEGEN" else lookup_address(token)
        if addr:
            await safe_tweet(
                text=build_dex_reply(addr),
                in_reply_to_tweet_id=tw.id
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

    # 4) general fallback
    prompt = (
        f"History:{history}\n"
        f"User asked: \"{txt}\"\n"
        "First, answer naturally and concisely. "
        "Then, in a second gambler-style line, mention stacking $DEGEN. End with NFA."
    )
    raw = ask_grok(prompt)
    
    # Ensure we have a complete response that doesn't get cut off
    reply_body = raw.strip()
    
    # Make sure the response contains $DEGEN mention and contract address
    if "$DEGEN" not in reply_body:
        reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
    else:
        # If $DEGEN is already mentioned, just add the contract address if needed
        if DEGEN_ADDR not in reply_body:
            reply = f"{reply_body}\n\nContract Address: {DEGEN_ADDR}"
        else:
            reply = reply_body
    
    # Ensure we're not exceeding Twitter's character limit
    if len(reply) > 260:
        reply = truncate_to_sentence(reply, 220) + f"\n\nContract Address: {DEGEN_ADDR}"
    
    img = choice(glob.glob("raid_images/*.jpg"))
    media_id = x_api.media_upload(img).media_id_string
    
    await safe_tweet(
        text=reply,
        media_id=media_id,
        in_reply_to_tweet_id=tw.id
    )
    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
    update_thread(convo_id, txt, reply)
    increment_thread(convo_id)

async def mention_loop():
    while True:
        try:
            last = redis_client.get(f"{REDIS_PREFIX}last_mention_id")
            params = {
                "id": BOT_ID,
                "tweet_fields": ["id", "text", "conversation_id"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            if last:
                params["since_id"] = int(last)
            res = await safe_mention_lookup(x_client.get_users_mentions, **params)
            if res and res.data:
                for tw in reversed(res.data):
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    redis_client.set(f"{REDIS_PREFIX}last_mention_id", tw.id)
                    await handle_mention(tw)
        except Exception as e:
            logger.error(f"Mention loop error: {e}")
        await asyncio.sleep(110)

async def search_mentions_loop():
    """
    New loop to handle searching for mentions that might not be captured by the mentions API,
    especially mentions in communities.
    """
    # Initialize last_search_id if not present
    if not redis_client.exists(f"{REDIS_PREFIX}last_search_id"):
        redis_client.set(f"{REDIS_PREFIX}last_search_id", INITIAL_SEARCH_ID)
        logger.info(f"Initialized last_search_id to {INITIAL_SEARCH_ID}")
    
    while True:
        try:
            query = f"@{BOT_USERNAME} -is:retweet"
            search_params = {
                "query": query,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "expansions": ["author_id"],
                "user_fields": ["username"],
                "max_results": 10
            }
            
            try:
                # Always try without since_id first to avoid the age restriction error
                search_results = await safe_search(
                    x_client.search_recent_tweets,
                    **search_params
                )
                logger.info("Successfully searched for community mentions")
            except Exception as e:
                logger.error(f"Search call failed: {e}", exc_info=True)
                search_results = None
            
            if search_results and search_results.data:
                newest_id = max(int(tw.id) for tw in search_results.data) if search_results.data else 0
                
                for tw in search_results.data:
                    # Skip if we've already processed this tweet
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    # Get the full tweet text from tw
                    logger.info(f"Processing community mention: {tw.id} - {tw.text[:30]}...")
                    
                    # Process the mention
                    await handle_mention(tw)
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                # Update the last search ID
                if newest_id > 0:
                    redis_client.set(f"{REDIS_PREFIX}last_search_id", str(newest_id))
                    logger.info(f"Updated last_search_id to {newest_id}")
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        # Wait before next search
        await asyncio.sleep(180)  # Run every 3 minutes

async def hourly_post_loop():
    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            raw = ask_grok("Write a one-sentence bullpost update on $DEGEN. Be promotional.")
            tweet = truncate_to_sentence(metrics + raw, 560)
            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                img = choice(glob.glob("raid_images/*.jpg"))
                media_id = x_api.media_upload(img).media_id_string
                await safe_tweet(tweet, media_id=media_id)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(mention_loop(), search_mentions_loop(), hourly_post_loop())

if __name__ == "__main__":
    asyncio.run(main())