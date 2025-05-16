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
    "PERPLEXITY_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
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
token_data = x_client.get_me().data
ASKDEGEN_ID = token_data.id
logger.info(f"Authenticated as: {token_data.username}, ID: {ASKDEGEN_ID}")

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
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
REDIS_CACHE_PREFIX = "degen:"
DEGEN_ADDRESS = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"

# Helper: call Perplexity API
def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    payload = {
        'model': 'sonar-pro',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': 1.0,
        'top_p': 0.9,
        'search_recency_filter': 'week'
    }
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json'
    }
    resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    return body['choices'][0]['message']['content'].strip()

# Fetch live data from DexScreener
def fetch_dexscreener_data(address: str) -> dict:
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except:
        pass
    resp = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
    resp.raise_for_status()
    data = resp.json()[0]
    result = {
        'symbol': data['baseToken']['symbol'],
        'price_usd': float(data.get('priceUsd', 0)),
        'liquidity_usd': float(data.get('liquidity', {}).get('usd', 0)),
        'volume_usd': float(data.get('volume', {}).get('h24', 0)),
        'market_cap': float(data.get('marketCap', 0))
    }
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result

# Resolve token symbol or address (unchanged)
def resolve_token(query: str) -> tuple:
    q = query.strip().upper()
    if q in ['DEGEN', '$DEGEN']:
        return 'DEGEN', DEGEN_ADDRESS
    if re.match(r'^[A-Za-z0-9]{43,44}$', q):
        return None, q
    system = (
        'Crypto analyst: map a Solana token symbol to its smart contract address. Return JSON: {"symbol": str, "address": str}.'
    )
    user = f"Symbol: {q}"
    out = ask_perplexity(system, user, max_tokens=60)
    try:
        obj = json.loads(out)
        if obj.get('address'):
            return obj.get('symbol'), obj.get('address')
    except:
        pass
    # Fallback skipped for brevity
    return None, None

# Handle a mention event
async def handle_mention(data: dict):
    evt = data['tweet_create_events'][0]
    txt = evt.get('text', '').replace('@askdegen', '').strip()
    tid = evt.get('id_str')

    # Case 1: Degen Confession
    if txt.lower().startswith('degen confession:'):
        reply = ask_perplexity(
            'Witty degen summarizer: anonymize and condense confession ≤750 chars.',
            txt,
            max_tokens=200
        )

    # Case 2: $TOKEN or on-chain address
    elif txt.startswith('$') or re.search(r'^[A-Za-z0-9]{43,44}$', txt):
        _, address = resolve_token(txt.split()[0])
        if not address:
            reply = 'Could not resolve token address.'
        else:
            data = fetch_dexscreener_data(address)
            # Build bullet list of data points
            bullets = [
                f"- Symbol: {data['symbol']}",
                f"- Price USD: ${data['price_usd']:.6f}",
                f"- 24h Volume USD: ${data['volume_usd']:.2f}",
                f"- Liquidity USD: ${data['liquidity_usd']:.2f}",
                f"- Market Cap: ${data['market_cap']:.2f}"
            ]
            bullet_text = '\n'.join(bullets)

            # Ask Perplexity for expert market view only
            analysis_prompt = (
                'Provide 2-3 sentences expert analysis of the current market trends ' 
                'for this Solana meme coin based on the above data. Do not repeat data points or mention the address.'
            )
            analysis = ask_perplexity(analysis_prompt, bullet_text, max_tokens=100)
            reply = f"{bullet_text}\n\n{analysis}"

    # Case 3: Freeform via Perplexity
    else:
        reply = ask_perplexity(
            'AtlasAI Degen Bot: professional assistant with dry degen humor—answer conversationally using real-time X data.',
            txt,
            max_tokens=150
        )

    # Tweet the reply
    x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
    logger.info(f"Replied to {tid}: {reply}")
    return {'message': 'Success'}, 200

# Polling loop to check mentions (unchanged)
async def poll_mentions():
    last_id = redis_client.get(f"{REDIS_CACHE_PREFIX}last_tweet_id")
    since_id = int(last_id) if last_id else None
    tweets = x_client.get_users_mentions(
        id=ASKDEGEN_ID,
        since_id=since_id,
        tweet_fields=['id', 'text', 'author_id'],
        expansions=['author_id'],
        user_fields=['username'],
        max_results=10
    )
    if not tweets or not tweets.data:
        return
    users = {u.id: u.username for u in tweets.includes.get('users', [])}
    for tw in reversed(tweets.data):
        user = users.get(tw.author_id, 'unknown')
        event = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': user}}]}
        try:
            await handle_mention(event)
        except Exception as e:
            logger.error(f'Error handling mention: {e}')
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_tweet_id", tw.id)
        redis_client.set(f"{REDIS_CACHE_PREFIX}last_mention", int(time.time()))

async def poll_mentions_loop():
    while True:
        await poll_mentions()
        lm = redis_client.get(f"{REDIS_CACHE_PREFIX}last_mention")
        interval = 90 if (lm and time.time() - int(lm) < 3600) else 1800
        logger.info(f"Sleeping {interval}s until next poll")
        await asyncio.sleep(interval)

async def reset_daily_counters():
    key = f"{REDIS_CACHE_PREFIX}last_reset"
    last = redis_client.get(key)
    now = time.time()
    target = time.mktime(time.strptime(f"{time.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")) - (4 * 3600)
    if not last or int(last) < target:
        redis_client.set(f"{REDIS_CACHE_PREFIX}read_count", 0)
        redis_client.set(f"{REDIS_CACHE_PREFIX}post_count", 0)
        redis_client.set(key, int(now))

@app.on_event('startup')
async def startup_event():
    await reset_daily_counters()
    asyncio.create_task(poll_mentions_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(request: Request):
    body = await request.json()
    txt = body.get('text', '@askdegen hello')
    event = {'tweet_create_events': [{'id_str': '0', 'text': txt, 'user': {'screen_name': 'test'}}]}
    return await handle_mention(event)
