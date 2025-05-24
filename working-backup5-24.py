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
import http.client
import sys
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of problematic tweet IDs to always skip
BLOCKED_TWEET_IDS = ["1924845778821845267"]

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

# Optional environment variables
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
MINIMUM_BUY_SOL = 1.0

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

try:
    me = x_client.get_me().data
    BOT_ID = me.id
    BOT_USERNAME = me.username
    logger.info(f"Authenticated as: {BOT_USERNAME} (ID: {BOT_ID})")
except Exception as e:
    logger.error(f"Authentication failed: {e}")
    logger.info("Check your API keys and upgrade to Basic tier ($100/month) minimum")
    exit(1)

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
USERNAME_RE = re.compile(rf"@{BOT_USERNAME}\b", re.IGNORECASE)

RATE_WINDOW = 900
MENTIONS_LIMIT = 8   # Reduced from 10
TWEETS_LIMIT = 25    # Reduced from 50
SEARCH_LIMIT = 15    # Reduced from 20
mentions_timestamps = deque()
tweet_timestamps = deque()
search_timestamps = deque()

# Set initial search ID to current time-based ID to avoid the "since_id too old" error
current_time_ms = int(time.time() * 1000) - 1728000000
INITIAL_SEARCH_ID = str((current_time_ms << 22))

# SEARCH TERMS - Keep your original targeting
SEARCH_QUERIES = [
    "memecoin -is:retweet -is:reply",
    "meme coin -is:retweet -is:reply", 
    "shitcoin -is:retweet -is:reply",
    "altcoin gem -is:retweet -is:reply",
    "moonshot -is:retweet -is:reply",
    "crypto pump -is:retweet -is:reply",
    "solana gem -is:retweet -is:reply",
    "#memecoin -is:retweet -is:reply",
    "#altcoin -is:retweet -is:reply",
    "buy the dip -is:retweet -is:reply",
    "diamond hands -is:retweet -is:reply",
    "hodl -is:retweet -is:reply",
    "$DOGE OR $SHIB OR $PEPE -is:retweet -is:reply",
    "$BONK OR $WIF OR $FLOKI -is:retweet -is:reply",
    "ape into -is:retweet -is:reply",
    "crypto twitter -is:retweet -is:reply",
    "good entry -is:retweet -is:reply",
    "accumulating -is:retweet -is:reply",
    "bullish on -is:retweet -is:reply",
    "next gem -is:retweet -is:reply",
    "x100 -is:retweet -is:reply",
    "to the moon -is:retweet -is:reply",
    "paper hands -is:retweet -is:reply",
    "wagmi -is:retweet -is:reply",
    "fud -is:retweet -is:reply"
]

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

def clean_reply_text(text: str) -> str:
    """Clean reply text properly - remove @mentions and fix punctuation"""
    # Remove @username patterns
    cleaned = re.sub(r'@\w+', '', text)
    
    # Fix common punctuation issues from @mention removal
    cleaned = re.sub(r'^[\s,\.\!]+', '', cleaned)  # Remove leading punctuation
    cleaned = re.sub(r'[\s,]+([,\.])', r'\1', cleaned)  # Fix double punctuation  
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Clean multiple spaces
    cleaned = cleaned.strip()
    
    # Ensure it starts with a capital letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned

def fix_degen_spacing(text: str) -> str:
    """Ensure $DEGEN has proper spacing for cashtag to work"""
    # Fix common spacing issues around $DEGEN
    text = re.sub(r'([^\s])(\$DEGEN)', r'\1 \2', text)  # Add space before if missing
    text = re.sub(r'(\$DEGEN)([^\s\.\,\!\?])', r'\1 \2', text)  # Add space after if missing  
    text = re.sub(r'-\s*\$DEGEN', r' $DEGEN', text)  # Fix "-$DEGEN" to " $DEGEN"
    return text

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

# Grok prompt - KEEP YOUR ORIGINAL PERSONALITY
SYSTEM_PROMPT = (
    "You are a degenerate gambler crypto analyst: edgy, informal, risk-taking. No slang. High class but a little edgy like Don Draper. "
    "IF DEGEN, or $DEGEN is mentioned, always ONLY speak about the $DEGEN token at contract address {DEGEN_ADDR}. DO not include the contract address in the reply. "
    "Do NOT mention any other token or chain when it comes to DEGEN. Other tokens you can reply honestly."
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
        wait_time = RATE_WINDOW - (now - timestamps_queue[0]) + 1
        logger.warning(f"Rate limit reached, waiting {wait_time:.0f}s...")
        await asyncio.sleep(wait_time)
    try:
        result = fn(*args, **kwargs)
        timestamps_queue.append(time.time())
        return result
    except (requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
        logger.warning(f"Network error during API call: {e}. Retrying in 5s…")
        await asyncio.sleep(5)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time()+RATE_WINDOW))
        wait_time = reset - time.time() + 1
        logger.warning(f"Rate limited, waiting {wait_time:.0f}s...")
        await asyncio.sleep(wait_time)
        return await safe_api_call(fn, timestamps_queue, limit, *args, **kwargs)
    except tweepy.Forbidden as e:
        logger.error(f"FORBIDDEN - Possible suspension: {e}")
        raise e
    except tweepy.BadRequest as e:
        logger.error(f"Bad request: {e}")
        raise e
    except Exception as e:
        logger.error(f"API call error: {e}", exc_info=True)
        raise e

async def safe_mention_lookup(fn, *args, **kwargs):
    return await safe_api_call(fn, mentions_timestamps, MENTIONS_LIMIT, *args, **kwargs)

async def safe_search(fn, *args, **kwargs):
    return await safe_api_call(fn, search_timestamps, SEARCH_LIMIT, *args, **kwargs)

async def safe_tweet(text: str, media_id=None, **kwargs):
    # CLEAN ALL @MENTIONS FROM TEXT AND FIX $DEGEN SPACING
    cleaned_text = clean_reply_text(text)
    cleaned_text = fix_degen_spacing(cleaned_text)
    
    return await safe_api_call(
        lambda t, m, **kw: x_client.create_tweet(text=t, media_ids=[m] if m else None, **kw),
        tweet_timestamps, 
        TWEETS_LIMIT,
        cleaned_text, 
        media_id, 
        **kwargs
    )

# DEX helpers - Keep your original functions
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
    """Format DEX data EXACTLY like the image - with proper spacing"""
    if not d:
        return "🚀 DEGEN | Data loading...\nMC Loading... | Vol24 Loading...\n1h Loading... | 24h Loading..."
    
    return (
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}\n"
        f"MC ${d['market_cap']:,.0f} | Vol24 ${d['volume_usd']:,.0f}\n"
        f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | "
        f"24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"
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
    """Build DEX reply with SAME FORMAT as hourly posts"""
    data = fetch_data(addr)
    if not data:
        return f"Data temporarily unavailable\n\nhttps://dexscreener.com/solana/{addr}"
    
    metrics = format_metrics(data)
    return f"{metrics}\n\n{data['link']}"

async def post_raid(tweet):
    """Post contextual $DEGEN promotion - KEEP YOUR ORIGINAL PROMPTS"""
    try:
        convo_id = tweet.conversation_id or tweet.id
        
        # Get author info if available
        author_info = ""
        if hasattr(tweet, 'author_id'):
            try:
                user_info = x_client.get_user(id=tweet.author_id)
                if user_info and user_info.data:
                    author_info = f" (from @{user_info.data.username})"
            except:
                pass
        
        # Create contextual $DEGEN promotion based on the tweet content - KEEP YOUR ORIGINAL PROMPTS
        tweet_lower = tweet.text.lower()
        
        if any(term in tweet_lower for term in ["memecoin", "meme coin", "shitcoin"]):
            prompt = f"User{author_info} posted about memecoins: '{tweet.text}'\nReply promoting $DEGEN as a solid memecoin play. Include contract {DEGEN_ADDR} and tag @ogdegenonsol. End with NFA. Be engaging and bullish like Don Draper."
        elif any(term in tweet_lower for term in ["gem", "moonshot", "x100"]):
            prompt = f"User{author_info} is looking for gems: '{tweet.text}'\nReply suggesting $DEGEN as a potential gem. Include contract {DEGEN_ADDR} and tag @ogdegenonsol. End with NFA. Be confident like Don Draper."
        elif any(term in tweet_lower for term in ["pump", "moon", "lambo"]):
            prompt = f"User{author_info} posted about pumps: '{tweet.text}'\nReply mentioning $DEGEN's potential. Include contract {DEGEN_ADDR} and tag @ogdegenonsol. End with NFA. Be optimistic like Don Draper."
        elif any(coin in tweet.text.upper() for coin in ["$DOGE", "$SHIB", "$PEPE", "$BONK"]):
            prompt = f"User{author_info} mentioned other memecoins: '{tweet.text}'\nReply suggesting they also check out $DEGEN. Include contract {DEGEN_ADDR} and tag @ogdegenonsol. End with NFA. Be friendly like Don Draper."
        else:
            prompt = f"User{author_info} posted about crypto: '{tweet.text}'\nReply naturally promoting $DEGEN. Include contract {DEGEN_ADDR} and tag @ogdegenonsol. End with NFA. Be engaging like Don Draper."
        
        msg = ask_grok(prompt)
        
        # Clean any existing contract address formatting and always add at end
        msg = msg.replace(f"CA: {DEGEN_ADDR}", "").replace(f"ca: {DEGEN_ADDR}", "").strip()
        msg = msg.replace(f"\n{DEGEN_ADDR}", "").replace(DEGEN_ADDR, "").strip()
        
        # Always add contract address cleanly at the end
        msg = f"{msg}\n\nCA: {DEGEN_ADDR}"
        
        # Try to use meme images
        media_id = None
        try:
            meme_files = glob.glob("raid_images/*.jpg")
            if meme_files:
                img = choice(meme_files)
                media_id = x_api.media_upload(img).media_id_string
        except tweepy.Forbidden as e:
            logger.warning(f"Media upload restricted, posting text-only: {e}")
            media_id = None
        except Exception as e:
            logger.warning(f"Media upload failed, posting text-only: {e}")
            media_id = None
        
        await safe_tweet(
            text=truncate_to_sentence(msg, 240),
            media_id=media_id,
            in_reply_to_tweet_id=tweet.id
        )
        
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))
        logger.info(f"✅ Posted crypto raid reply to tweet {tweet.id}")
        
        # Random delay to look more human
        await asyncio.sleep(random.uniform(10, 30))
        
    except Exception as e:
        logger.error(f"Error in post_raid for tweet {tweet.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tweet.id))

async def search_degen_loop():
    """Search for 'degen' mentions and raid them"""
    key = f"{REDIS_PREFIX}last_degen_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)

    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "degen -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text", "conversation_id", "created_at"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                for tw in res.data:
                    tid = str(tw.id)
                    if tid in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", tid):
                        continue
                    await post_raid(tw)
                    redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tid)
                redis_client.set(key, str(newest))
                logger.info(f"🎯 Processed {len(res.data)} degen mentions")
        except Exception as e:
            logger.error(f"search_degen_loop error: {e}", exc_info=True)
        await asyncio.sleep(300)  # every 5 minutes

async def broad_crypto_raid_loop():
    """Crypto raiding - REDUCED RATE for sustainability"""
    query_index = 0
    
    while True:
        try:
            # Rotate through different search queries for maximum coverage
            current_query = SEARCH_QUERIES[query_index % len(SEARCH_QUERIES)]
            query_index += 1
            
            params = {
                "query": current_query,
                "tweet_fields": ["id", "text", "conversation_id", "created_at", "author_id"],
                "expansions": ["author_id"],
                "user_fields": ["username", "public_metrics"],
                "max_results": 25
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            
            if res and res.data:
                # Create user mapping
                user_map = {}
                if hasattr(res, 'includes') and res.includes and 'users' in res.includes:
                    for user in res.includes['users']:
                        user_map[user.id] = user
                
                qualified_tweets = []
                for tw in res.data:
                    tid = str(tw.id)
                    
                    # Skip blocked tweets and already replied tweets
                    if tid in BLOCKED_TWEET_IDS or redis_client.sismember(f"{REDIS_PREFIX}replied_ids", tid):
                        continue
                    
                    # Get author info
                    author = user_map.get(tw.author_id)
                    follower_count = 0
                    if author and hasattr(author, 'public_metrics'):
                        follower_count = author.public_metrics.get('followers_count', 0)
                    
                    # QUALITY TARGETING - focus on engaged accounts
                    should_raid = False
                    
                    # 1. Prioritize accounts with decent following (30+ followers) 
                    if follower_count >= 30:
                        should_raid = True
                        
                    # 2. Always raid tweets mentioning other memecoins (perfect audience)
                    elif any(coin in tw.text.upper() for coin in ["$DOGE", "$SHIB", "$PEPE", "$BONK", "$WIF"]):
                        should_raid = True
                        
                    # 3. Raid substantial crypto discussions from smaller accounts
                    elif follower_count >= 15 and len(tw.text) > 60 and any(term in tw.text.lower() for term in ["crypto", "coin", "token", "blockchain"]):
                        should_raid = True
                    
                    if should_raid:
                        qualified_tweets.append(tw)
                        username = author.username if author else 'unknown'
                        logger.info(f"🎯 CRYPTO RAID: @{username} ({follower_count} followers): {tw.text[:50]}...")
                
                # Process FEWER qualified tweets - SUSTAINABLE RATE
                for tw in qualified_tweets[:3]:  # Only 3 per cycle
                    try:
                        await post_raid(tw)
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                    except Exception as e:
                        logger.error(f"Error processing crypto raid {tw.id}: {e}")
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                logger.info(f"🚀 CRYPTO RAIDED {len(qualified_tweets[:3])} tweets using query: '{current_query[:30]}...'")
                
            else:
                logger.info(f"🔍 No results for query: '{current_query[:30]}...'")
                
        except Exception as e:
            logger.error(f"broad_crypto_raid_loop error: {e}", exc_info=True)
        
        await asyncio.sleep(600)  # Every 10 minutes - SUSTAINABLE

async def auto_like_degen_loop():
    """Like tweets mentioning $DEGEN"""
    key = f"{REDIS_PREFIX}last_like_id"
    if not redis_client.exists(key):
        redis_client.set(key, INITIAL_SEARCH_ID)

    while True:
        try:
            last_id = redis_client.get(key)
            params = {
                "query": "DEGEN -is:retweet",
                "since_id": last_id,
                "tweet_fields": ["id", "text"],
                "max_results": 10
            }
            res = await safe_search(x_client.search_recent_tweets, **params)
            if res and res.data:
                newest = max(int(t.id) for t in res.data)
                liked_count = 0
                for tw in res.data:
                    tid = str(tw.id)
                    if "$DEGEN" in tw.text.upper() and not redis_client.sismember(f"{REDIS_PREFIX}liked_ids", tid):
                        try:
                            x_api.create_favorite(id=tid)
                            redis_client.sadd(f"{REDIS_PREFIX}liked_ids", tid)
                            liked_count += 1
                            logger.info(f"👍 Liked $DEGEN tweet: {tid}")
                        except Exception as e:
                            logger.warning(f"Like failed for {tid}: {e}")
                            redis_client.sadd(f"{REDIS_PREFIX}liked_ids", tid)
                redis_client.set(key, str(newest))
                logger.info(f"💙 Liked {liked_count} $DEGEN tweets")
        except Exception as e:
            logger.error(f"auto_like_degen_loop error: {e}", exc_info=True)
        await asyncio.sleep(300)  # every 5 minutes

async def handle_mention(tw):
    """FIXED MENTION HANDLING - Clean replies with contract address and memes"""
    try:
        convo_id = tw.conversation_id or tw.id
        if redis_client.hget(get_thread_key(convo_id), "count") is None:
            try:
                root = x_client.get_tweet(convo_id, tweet_fields=['text']).data.text
                update_thread(convo_id, f"ROOT: {root}", "")
            except Exception as e:
                logger.warning(f"Failed to get root tweet: {e}")
                update_thread(convo_id, f"ROOT: Unknown", "")
        
        history = get_thread_history(convo_id)
        txt = re.sub(rf"@{BOT_USERNAME}\b", "", tw.text, flags=re.IGNORECASE).strip()

        # 1) raid command
        if re.search(r"\braid\b", txt, re.IGNORECASE):
            await post_raid(tw)
            return

        # 2) Check for CA command (contract address only)
        if re.search(r"\bca\b", txt, re.IGNORECASE) and not re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=f"$DEGEN Contract Address: {DEGEN_ADDR}",
                in_reply_to_tweet_id=tw.id
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 3) Check for DEX or other contract address commands
        if re.search(r"\b(dex|contract|address)\b", txt, re.IGNORECASE):
            await safe_tweet(
                text=f"{build_dex_reply(DEGEN_ADDR)}",
                in_reply_to_tweet_id=tw.id
            )
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
            return

        # 4) token/address -> DEX preview
        token = next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)), None)
        if token:
            sym = token.lstrip('$').upper()
            addr = DEGEN_ADDR if sym=="DEGEN" else lookup_address(token)
            if addr:
                await safe_tweet(
                    text=build_dex_reply(addr),
                    in_reply_to_tweet_id=tw.id
                )
                redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                return

        # 5) general fallback - CLEAN REPLY WITH CONTRACT ADDRESS
        prompt = (
            f"History:{history}\n"
            f"User asked: \"{txt}\"\n"
            "Answer naturally and concisely."
        )
        raw = ask_grok(prompt)
        
        reply_body = raw.strip()
        
        # Always include $DEGEN and contract address
        if "$DEGEN" not in reply_body:
            reply = f"{reply_body}\n\nStack $DEGEN! Contract Address: {DEGEN_ADDR}"
        else:
            if DEGEN_ADDR not in reply_body:
                reply = f"{reply_body}\n\nStack $DEGEN. CA: {DEGEN_ADDR}"
            else:
                reply = reply_body
        
        if len(reply) > 360:
            reply = truncate_to_sentence(reply, 360) + f"\n\n$DEGEN. CA: {DEGEN_ADDR}"
        
        # Try to use meme image
        media_id = None
        try:
            meme_files = glob.glob("raid_images/*.jpg")
            if meme_files:
                img = choice(meme_files)
                media_id = x_api.media_upload(img).media_id_string
        except Exception as e:
            logger.warning(f"Meme upload failed: {e}")
            media_id = None
        
        await safe_tweet(
            text=reply,
            media_id=media_id,
            in_reply_to_tweet_id=tw.id
        )
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
        update_thread(convo_id, txt, reply)
        increment_thread(convo_id)
        
    except Exception as e:
        logger.error(f"Error handling mention {tw.id}: {e}", exc_info=True)
        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))

async def search_mentions_loop():
    """Search for mentions that might not be captured by the mentions API"""
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
                    if str(tw.id) in BLOCKED_TWEET_IDS:
                        logger.info(f"Skipping blocked tweet ID: {tw.id}")
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                        continue
                        
                    if redis_client.sismember(f"{REDIS_PREFIX}replied_ids", str(tw.id)):
                        continue
                    
                    try:
                        logger.info(f"Processing community mention: {tw.id} - {tw.text[:30]}...")
                        await handle_mention(tw)
                    except Exception as e:
                        logger.error(f"Error processing mention {tw.id}: {e}", exc_info=True)
                        redis_client.sadd(f"{REDIS_PREFIX}replied_ids", str(tw.id))
                
                if newest_id > 0:
                    redis_client.set(f"{REDIS_PREFIX}last_search_id", str(newest_id))
                    logger.info(f"Updated last_search_id to {newest_id}")
                    
        except Exception as e:
            logger.error(f"Search mentions loop error: {e}", exc_info=True)
        
        await asyncio.sleep(300)  # Every 5 minutes

async def hourly_post_loop():
    """HOURLY POSTS - EXACT FORMAT MATCHING THE IMAGE"""
    grok_prompts = [
        "Write a positive one-sentence analytical update on $DEGEN using data from the last hour. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace.",
        "Write a positive one-sentence cryptic message about secret tech being developed on $DEGEN's price action. Be edgy and risky. Do not mention the contract address. No slang. High class but a little edgy like Don Draper.",
        "Write a one sentence, cryptic message about $DEGEN that implies insider knowledge. Do not mention the contract address. No slang. High class but a little edgy like David Foster Wallace.",
        "Write a one sentence, cryptic comment about people who haven't bought $DEGEN yet. Do not mention the contract address. No slang. High class but a little edgy like Elon Musk.",
        "Write a one sentence comparing $DEGEN to the broader crypto market. Be cryptic. Do not mention the contract address. No slang. High class but a little edgy like Hemingway.",
        "Write a one sentence post about diamond hands and $DEGEN's future potential. Do not mention the contract address. No slang. High class but a little edgy like Hunter Thompson."
    ]
    
    hour_counter = 0

    while True:
        try:
            data = fetch_data(DEGEN_ADDR)
            metrics = format_metrics(data)
            dex_link = data.get('link', f"https://dexscreener.com/solana/{DEGEN_ADDR}")

            selected_prompt = grok_prompts[hour_counter % len(grok_prompts)]
            raw = ask_grok(selected_prompt).strip()

            # EXACT FORMAT FROM THE IMAGE:
            # 1. DEX metrics (with icons)
            # 2. Blank line
            # 3. Grok sentence  
            # 4. Blank line
            # 5. Link (for preview image)
            tweet = f"{metrics}\n\n{raw}\n\n{dex_link}"

            last = redis_client.get(f"{REDIS_PREFIX}last_hourly_post")
            if tweet != last:
                await safe_tweet(tweet)
                redis_client.set(f"{REDIS_PREFIX}last_hourly_post", tweet)
                logger.info("Posted hourly update")

            hour_counter += 1
        except Exception as e:
            logger.error(f"Hourly post error: {e}")
        
        await asyncio.sleep(3600)

async def main():
    try:
        logger.info("🚀 Starting CRYPTO PROMOTION bot for $DEGEN...")
        logger.info("✅ Fixed: Clean replies, contract addresses, meme images, proper hourly format")
        
        # Pre-mark all blocked tweets as replied to
        for tweet_id in BLOCKED_TWEET_IDS:
            redis_client.sadd(f"{REDIS_PREFIX}replied_ids", tweet_id)
            logger.info(f"Pre-marked blocked tweet ID {tweet_id} as replied")
        
        logger.info("💎 Starting all bot functions...")
        
        # Run all loops
        await asyncio.gather(
            search_mentions_loop(),      # Handle @mentions with clean replies
            hourly_post_loop(),         # EXACT format as image 
            search_degen_loop(),        # Search 'degen' and raid
            broad_crypto_raid_loop(),   # Broad crypto raiding
            auto_like_degen_loop(),     # Like $DEGEN tweets
        )
        
    except Exception as e:
        logger.error(f"Main function error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)