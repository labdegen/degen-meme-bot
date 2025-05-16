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

# Load env vars and validate
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "PERPLEXITY_API_KEY",  # Added Perplexity API key
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
optional_vars = ["BIRDEYE_API_KEY"]  # Optional API keys
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")
        
for v in optional_vars:
    if not os.getenv(v):
        logger.warning(f"Optional env var missing: {v}")


# API endpoints
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Credentials
API_KEY = os.getenv("X_API_KEY")
API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
GROK_KEY = os.getenv("GROK_API_KEY")
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
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE = re.compile(r'^[A-Za-z0-9]{43,44}$')
TICKER_RE = re.compile(r'\$([A-Za-z0-9]{1,10})')

# Helper functions
def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {GROK_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Grok fallback: {e}")
        return "Insight currently unavailable. Check back soon."

def ask_perplexity(query: str, token_info: dict = None, max_tokens: int = 200) -> str:
    system_prompt = """You are a professional crypto analyst providing brief insights on tokens.
If given market data, incorporate it into your analysis. Focus on recent news, market sentiment, and trading patterns.
Never mention that you're an AI or that your knowledge has limitations. Be clear, insightful and stay under 150 characters."""
    
    if token_info:
        user_prompt = f"Analyze {token_info['symbol']} (${token_info['price_usd']:.6f}) with 24h change of {token_info['change_24h']:.2f}%. {query}"
    else:
        user_prompt = query
        
    body = {
        "model": "sonar-small-online",  # Using Perplexity's real-time search model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(PERPLEXITY_URL, json=body, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Perplexity fallback: {e}")
        return None  # Will fall back to Grok

def fetch_data(addr: str) -> dict:
    cache_key = f"{REDIS_PREFIX}dex:{addr}"
    if cached := db.get(cache_key):
        return json.loads(cached)
    try:
        r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
        r.raise_for_status()
        pairs = r.json()
        if not pairs:
            logger.warning(f"No data found for address: {addr}")
            return None
            
        data = pairs[0]
        base = data.get('baseToken', {})
        out = {
            'symbol':     base.get('symbol'),
            'price_usd':  float(data.get('priceUsd', 0)),
            'volume_usd': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_1h':  float(data.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link':       f"https://dexscreener.com/solana/{addr}"
        }
        db.setex(cache_key, 300, json.dumps(out))
        return out
    except Exception as e:
        logger.error(f"Error fetching data for {addr}: {e}")
        return None

def resolve_token(q: str) -> tuple:
    """Improved token resolution with better ticker handling"""
    if not q:
        return None, None
        
    s = q.upper().lstrip('$')

def format_token_data(d: dict) -> str:
    """Format token data into a Twitter-friendly message"""
    if not d:
        return "Could not fetch token data at this time."
        
    lines = [
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
        f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
        f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
        d['link']
    ]
    return "\n".join(lines)

def extract_ticker(text: str) -> str:
    """Extract ticker symbol from text"""
    matches = TICKER_RE.findall(text)
    return matches[0] if matches else None

async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or not isinstance(events, list) or not events[0].get("text"):
        logger.warning("Skipping invalid or empty mention event")
        return {"message": "no valid mention"}

    tweet_text = events[0]['text'].replace('@askdegen', '').strip()
    tweet_id = events[0]['id_str']
    
    # Extract ticker from the tweet text
    ticker_from_regex = extract_ticker(tweet_text)
    
    # Find token either explicitly mentioned or in question
    explicit_token = next((w for w in tweet_text.split() if w.startswith('

async def degen_hourly_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            if not d:
                logger.error("Failed to fetch DEGEN data for hourly update")
                await asyncio.sleep(600)  # Try again in 10 minutes
                continue
                
            # Generate the stats card
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            
            # Try Perplexity first for latest news/insights
            perplexity_prompt = f"Latest news and market insights for DEGEN token on Solana. Current price: ${d['price_usd']:.6f}, 24h change: {d['change_24h']:+.2f}%. Keep it brief and actionable."
            analysis = ask_perplexity(perplexity_prompt, d, max_tokens=120)  # Reduced token count
            
            # Fall back to Grok if needed
            if not analysis:
                sys_msg = "You are a DEGEN community insider. Write a 2-sentence hourly update based on this data. Be enthusiastic but grounded. Do not mention Solana."
                analysis = ask_grok(sys_msg, json.dumps(d), max_tokens=120)  # Reduced token count
            
            # Calculate available characters for analysis
            # Fixed card size + 2 extra newlines
            card_size = sum(len(line) for line in card) + 3  # +3 for newlines
            max_analysis_chars = 275 - card_size  # 275 instead of 280 for safety margin
            
            # Truncate analysis if needed, ensuring we end on a complete sentence
            if len(analysis) > max_analysis_chars:
                analysis = analysis[:max_analysis_chars]
                # Find the last sentence boundary
                last_period = analysis.rfind('.')
                if last_period > max_analysis_chars * 0.6:  # Only truncate if we're not losing too much
                    analysis = analysis[:last_period+1]
                else:
                    # If no good sentence boundary, find last space
                    last_space = analysis.rfind(' ')
                    analysis = analysis[:last_space] + '...'
                
            # Combine card and analysis with proper spacing
            tweet = "\n".join(card + ["", analysis])
                    
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted")
            except Exception as e:
                logger.error(f"Error posting via client: {e}")
                try:
                    x_api.update_status(tweet)
                    logger.info("Hourly promo posted via API")
                except Exception as e2:
                    logger.error(f"Second error posting hourly update: {e2}")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
            
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        try:
            last = db.get(f"{REDIS_PREFIX}last_tweet_id")
            since_id = int(last) if last else None
            res = x_client.get_users_mentions(
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
                    ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                    try:
                        await handle_mention(ev)
                    except Exception as e:
                        logger.error(f"Mention error: {e}")
                    db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                    db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
            
            # Adjust polling frequency based on recent activity
            lm = db.get(f"{REDIS_PREFIX}last_mention")
            if lm and time.time() - int(lm) < 3600:
                # More frequent checks if there was activity in the last hour
                await asyncio.sleep(60)
            else:
                # Otherwise check less frequently
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
            await asyncio.sleep(300)  # Back off on error

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    data = await r.json()
    ev = {'tweet_create_events': [{'id_str': '0', 'text': data.get('text', ''), 'user': {'screen_name': 'test'}}]}
    return await handle_mention(ev)
)
    
    # Handle known tokens
    if s == 'DEGEN':
        return 'DEGEN', DEGEN_ADDR
        
    # Check if it's a direct address
    if ADDR_RE.match(s):
        return None, s
        
    # Try to find in cache first
    cache_key = f"{REDIS_PREFIX}token:{s}"
    if cached := db.get(cache_key):
        cached_data = json.loads(cached)
        return cached_data["symbol"], cached_data["address"]
    
    # Search on DEXScreener - first try
    try:
        logger.info(f"Searching for token: {s}")
        resp = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}", timeout=10)
        resp.raise_for_status()
        
        # First try exact match for Solana tokens
        for item in resp.json().get("pairs", []):
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                if base.get('symbol', '').upper() == s:
                    symbol = base.get('symbol')
                    addr = item.get('pairAddress') or base.get('address')
                    if addr:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
        
        # If no exact match, try any Solana token in results
        for item in resp.json().get("pairs", []):
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                symbol = base.get('symbol')
                addr = item.get('pairAddress') or base.get('address')
                if addr:
                    db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                    return symbol, addr
    except Exception as e:
        logger.error(f"Error in first DEXScreener search for {s}: {e}")
    
    # Try another approach for unknown tokens - search using Jupiter API
    try:
        logger.info(f"Trying Jupiter API for token: {s}")
        jupiter_url = "https://token.jup.ag/strict"
        resp = requests.get(jupiter_url, timeout=10)
        resp.raise_for_status()
        
        token_list = resp.json()
        # Look for exact match first
        for token in token_list:
            if token.get('symbol', '').upper() == s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
        
        # Try partial match if no exact match
        for token in token_list:
            if s in token.get('symbol', '').upper() or token.get('symbol', '').upper() in s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
    except Exception as e:
        logger.error(f"Error in Jupiter search for {s}: {e}")
        
    # Try one more approach - check Birdeye API
    try:
        logger.info(f"Trying Birdeye API for token: {s}")
        headers = {
            "X-API-KEY": os.getenv("BIRDEYE_API_KEY", ""),  # Optional API key
            "User-Agent": "Mozilla/5.0"
        }
        birdeye_url = f"https://public-api.birdeye.so/public/tokenlist/solana?search={s}"
        resp = requests.get(birdeye_url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        token_list = resp.json().get("data", [])
        # Look for exact match first
        for token in token_list:
            if token.get('symbol', '').upper() == s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
                        
        # Try partial match
        for token in token_list:
            if s in token.get('symbol', '').upper():
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
    except Exception as e:
        logger.error(f"Error in Birdeye search for {s}: {e}")
    
    return None, None

def format_token_data(d: dict) -> str:
    """Format token data into a Twitter-friendly message"""
    if not d:
        return "Could not fetch token data at this time."
        
    lines = [
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
        f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
        f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
        d['link']
    ]
    return "\n".join(lines)

def extract_ticker(text: str) -> str:
    """Extract ticker symbol from text"""
    matches = TICKER_RE.findall(text)
    return matches[0] if matches else None

async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or not isinstance(events, list) or not events[0].get("text"):
        logger.warning("Skipping invalid or empty mention event")
        return {"message": "no valid mention"}

    tweet_text = events[0]['text'].replace('@askdegen', '').strip()
    tweet_id = events[0]['id_str']
    
    # Extract ticker from the tweet text
    ticker_from_regex = extract_ticker(tweet_text)
    
    # Find token either explicitly mentioned or in question
    explicit_token = next((w for w in tweet_text.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    token = explicit_token or (f"${ticker_from_regex}" if ticker_from_regex else None)
    
    reply = None
    token_data = None

    if token:
        sym, addr = resolve_token(token)
        logger.info(f"Resolved {token} to symbol={sym}, address={addr}")
        
        if addr:
            token_data = fetch_data(addr)
            
            # If only token is mentioned, return just the data
            if tweet_text.strip() == token or len(tweet_text.split()) <= 2:
                reply = format_token_data(token_data)
            # Otherwise, we have a question about the token - use Perplexity with token data
            else:
                reply = ask_perplexity(tweet_text, token_data, max_tokens=160)
                if reply:
                    # Combine token data and Perplexity response
                    token_text = format_token_data(token_data)
                    reply = f"{token_text}\n\n{reply}"
                else:
                    # Fall back to Grok with token data
                    prompt = f"You are a professional crypto market analyst. Given: {json.dumps(token_data)}, reply to an investor in <140 characters about their query: {tweet_text}"
                    analysis = ask_grok(prompt, tweet_text, max_tokens=160)
                    reply = f"{format_token_data(token_data)}\n\n{analysis}"
        elif ticker_from_regex:
            # Try Perplexity for general information about the token
            query = f"What's the latest on ${ticker_from_regex} crypto token? Latest price, trends, and news in brief."
            reply = ask_perplexity(query, max_tokens=180)
        
    # If no token data and no reply yet, use Perplexity for general crypto questions
    if not reply and any(word in tweet_text.lower() for word in ["crypto", "token", "coin", "market", "price", "sol", "btc", "eth"]):
        reply = ask_perplexity(tweet_text, max_tokens=180)
        
    # Final fallback to Grok for any other questions
    if not reply:
        system_msg = "Professional crypto analyst: provide concise, insightful response under 240 characters."
        reply = ask_grok(system_msg, tweet_text, max_tokens=180)

    # Ensure tweet isn't too long
    tweet = reply.strip()
    if len(tweet) > 280:
        tweet = tweet[:280]
        if '.' in tweet:
            tweet = tweet.rsplit('.', 1)[0] + '.'
        else:
            tweet = tweet.rsplit(' ', 1)[0] + '...'
            
    # Post reply
    try:
        x_client.create_tweet(text=tweet, in_reply_to_tweet_id=int(tweet_id))
        logger.info(f"Posted reply to {tweet_id}")
    except Exception as e:
        logger.error(f"Error posting reply: {e}")
        try:
            x_api.update_status(status=tweet, in_reply_to_status_id=tweet_id)
            logger.info(f"Posted reply via API to {tweet_id}")
        except Exception as e2:
            logger.error(f"Second error posting reply: {e2}")
            
    return {'message': 'ok'}

async def degen_hourly_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            if not d:
                logger.error("Failed to fetch DEGEN data for hourly update")
                await asyncio.sleep(600)  # Try again in 10 minutes
                continue
                
            # Generate the stats card
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            
            # Try Perplexity first for latest news/insights
            perplexity_prompt = f"Latest news and market insights for DEGEN token on Solana. Current price: ${d['price_usd']:.6f}, 24h change: {d['change_24h']:+.2f}%. Keep it brief and actionable."
            analysis = ask_perplexity(perplexity_prompt, d, max_tokens=120)  # Reduced token count
            
            # Fall back to Grok if needed
            if not analysis:
                sys_msg = "You are a DEGEN community insider. Write a 2-sentence hourly update based on this data. Be enthusiastic but grounded. Do not mention Solana."
                analysis = ask_grok(sys_msg, json.dumps(d), max_tokens=120)  # Reduced token count
            
            # Calculate available characters for analysis
            # Fixed card size + 2 extra newlines
            card_size = sum(len(line) for line in card) + 3  # +3 for newlines
            max_analysis_chars = 275 - card_size  # 275 instead of 280 for safety margin
            
            # Truncate analysis if needed, ensuring we end on a complete sentence
            if len(analysis) > max_analysis_chars:
                analysis = analysis[:max_analysis_chars]
                # Find the last sentence boundary
                last_period = analysis.rfind('.')
                if last_period > max_analysis_chars * 0.6:  # Only truncate if we're not losing too much
                    analysis = analysis[:last_period+1]
                else:
                    # If no good sentence boundary, find last space
                    last_space = analysis.rfind(' ')
                    analysis = analysis[:last_space] + '...'
                
            # Combine card and analysis with proper spacing
            tweet = "\n".join(card + ["", analysis])
                    
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted")
            except Exception as e:
                logger.error(f"Error posting via client: {e}")
                try:
                    x_api.update_status(tweet)
                    logger.info("Hourly promo posted via API")
                except Exception as e2:
                    logger.error(f"Second error posting hourly update: {e2}")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
            
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        try:
            last = db.get(f"{REDIS_PREFIX}last_tweet_id")
            since_id = int(last) if last else None
            res = x_client.get_users_mentions(
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
                    ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                    try:
                        await handle_mention(ev)
                    except Exception as e:
                        logger.error(f"Mention error: {e}")
                    db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                    db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
            
            # Adjust polling frequency based on recent activity
            lm = db.get(f"{REDIS_PREFIX}last_mention")
            if lm and time.time() - int(lm) < 3600:
                # More frequent checks if there was activity in the last hour
                await asyncio.sleep(60)
            else:
                # Otherwise check less frequently
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
            await asyncio.sleep(300)  # Back off on error

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    data = await r.json()
    ev = {'tweet_create_events': [{'id_str': '0', 'text': data.get('text', ''), 'user': {'screen_name': 'test'}}]}
    return await handle_mention(ev)
) or ADDR_RE.match(w)), None)
    token = explicit_token or (f"${ticker_from_regex}" if ticker_from_regex else None)
    
    reply = None
    token_data = None

    if token:
        sym, addr = resolve_token(token)
        logger.info(f"Resolved {token} to symbol={sym}, address={addr}")
        
        if addr:
            token_data = fetch_data(addr)
            
            # If only token is mentioned, return just the data
            if tweet_text.strip() == token or len(tweet_text.split()) <= 2:
                if token_data:
                    reply = format_token_data(token_data)
                else:
                    # If we couldn't get data, try Perplexity to at least give token info
                    query = f"What is {token} crypto token? Give price if known and basic info, under 200 characters."
                    reply = ask_perplexity(query, max_tokens=180)
            # Otherwise, we have a question about the token - use Perplexity with token data
            else:
                if token_data:
                    perplexity_reply = ask_perplexity(tweet_text, token_data, max_tokens=160)
                    if perplexity_reply:
                        # Combine token data and Perplexity response
                        token_text = format_token_data(token_data)
                        reply = f"{token_text}\n\n{perplexity_reply}"
                    else:
                        # Fall back to Grok with token data
                        prompt = f"You are a professional crypto market analyst. Given: {json.dumps(token_data)}, reply to an investor in <140 characters about their query: {tweet_text}"
                        analysis = ask_grok(prompt, tweet_text, max_tokens=160)
                        reply = f"{format_token_data(token_data)}\n\n{analysis}"
                else:
                    # We have a symbol but no data, try Perplexity for general info
                    query = f"Answer this question about {token} crypto token: {tweet_text}"
                    reply = ask_perplexity(query, max_tokens=240)
        elif ticker_from_regex:
            # Try Perplexity for general information about the token
            query = f"What's the latest on ${ticker_from_regex} crypto token? Latest price, trends, and news in brief."
            reply = ask_perplexity(query, max_tokens=240)
        
    # If no token data and no reply yet, use Perplexity for general crypto questions
    if not reply and any(word in tweet_text.lower() for word in ["crypto", "token", "coin", "market", "price", "sol", "btc", "eth"]):
        reply = ask_perplexity(tweet_text, max_tokens=240)
        
    # Final fallback to Grok for any other questions
    if not reply:
        system_msg = "Professional crypto analyst: provide concise, insightful response under 240 characters."
        reply = ask_grok(system_msg, tweet_text, max_tokens=240)

    # Ensure tweet isn't too long
    tweet = reply.strip()
    if len(tweet) > 278: # leave a bit of margin
        # Find a good cutoff point - prefer complete sentences
        cutoff = 275
        last_period = tweet[:cutoff].rfind('.')
        
        if last_period > cutoff * 0.7:  # Only use period if it's reasonably far in
            tweet = tweet[:last_period+1]
        else:
            # Fall back to word boundaries
            cutoff_point = tweet[:cutoff].rfind(' ')
            tweet = tweet[:cutoff_point] + '...'
            
    # Post reply
    try:
        x_client.create_tweet(text=tweet, in_reply_to_tweet_id=int(tweet_id))
        logger.info(f"Posted reply to {tweet_id}")
    except Exception as e:
        logger.error(f"Error posting reply: {e}")
        try:
            x_api.update_status(status=tweet, in_reply_to_status_id=tweet_id)
            logger.info(f"Posted reply via API to {tweet_id}")
        except Exception as e2:
            logger.error(f"Second error posting reply: {e2}")
            
    return {'message': 'ok'}

async def degen_hourly_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            if not d:
                logger.error("Failed to fetch DEGEN data for hourly update")
                await asyncio.sleep(600)  # Try again in 10 minutes
                continue
                
            # Generate the stats card
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            
            # Try Perplexity first for latest news/insights
            perplexity_prompt = f"Latest news and market insights for DEGEN token on Solana. Current price: ${d['price_usd']:.6f}, 24h change: {d['change_24h']:+.2f}%. Keep it brief and actionable."
            analysis = ask_perplexity(perplexity_prompt, d, max_tokens=120)  # Reduced token count
            
            # Fall back to Grok if needed
            if not analysis:
                sys_msg = "You are a DEGEN community insider. Write a 2-sentence hourly update based on this data. Be enthusiastic but grounded. Do not mention Solana."
                analysis = ask_grok(sys_msg, json.dumps(d), max_tokens=120)  # Reduced token count
            
            # Calculate available characters for analysis
            # Fixed card size + 2 extra newlines
            card_size = sum(len(line) for line in card) + 3  # +3 for newlines
            max_analysis_chars = 275 - card_size  # 275 instead of 280 for safety margin
            
            # Truncate analysis if needed, ensuring we end on a complete sentence
            if len(analysis) > max_analysis_chars:
                analysis = analysis[:max_analysis_chars]
                # Find the last sentence boundary
                last_period = analysis.rfind('.')
                if last_period > max_analysis_chars * 0.6:  # Only truncate if we're not losing too much
                    analysis = analysis[:last_period+1]
                else:
                    # If no good sentence boundary, find last space
                    last_space = analysis.rfind(' ')
                    analysis = analysis[:last_space] + '...'
                
            # Combine card and analysis with proper spacing
            tweet = "\n".join(card + ["", analysis])
                    
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted")
            except Exception as e:
                logger.error(f"Error posting via client: {e}")
                try:
                    x_api.update_status(tweet)
                    logger.info("Hourly promo posted via API")
                except Exception as e2:
                    logger.error(f"Second error posting hourly update: {e2}")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
            
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        try:
            last = db.get(f"{REDIS_PREFIX}last_tweet_id")
            since_id = int(last) if last else None
            res = x_client.get_users_mentions(
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
                    ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                    try:
                        await handle_mention(ev)
                    except Exception as e:
                        logger.error(f"Mention error: {e}")
                    db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                    db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
            
            # Adjust polling frequency based on recent activity
            lm = db.get(f"{REDIS_PREFIX}last_mention")
            if lm and time.time() - int(lm) < 3600:
                # More frequent checks if there was activity in the last hour
                await asyncio.sleep(60)
            else:
                # Otherwise check less frequently
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
            await asyncio.sleep(300)  # Back off on error

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    data = await r.json()
    ev = {'tweet_create_events': [{'id_str': '0', 'text': data.get('text', ''), 'user': {'screen_name': 'test'}}]}
    return await handle_mention(ev)
)
    
    # Handle known tokens
    if s == 'DEGEN':
        return 'DEGEN', DEGEN_ADDR
        
    # Check if it's a direct address
    if ADDR_RE.match(s):
        return None, s
        
    # Try to find in cache first
    cache_key = f"{REDIS_PREFIX}token:{s}"
    if cached := db.get(cache_key):
        cached_data = json.loads(cached)
        return cached_data["symbol"], cached_data["address"]
    
    # Search on DEXScreener - first try
    try:
        logger.info(f"Searching for token: {s}")
        resp = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}", timeout=10)
        resp.raise_for_status()
        
        # First try exact match for Solana tokens
        for item in resp.json().get("pairs", []):
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                if base.get('symbol', '').upper() == s:
                    symbol = base.get('symbol')
                    addr = item.get('pairAddress') or base.get('address')
                    if addr:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
        
        # If no exact match, try any Solana token in results
        for item in resp.json().get("pairs", []):
            if item.get('chainId') == 'solana':
                base = item.get('baseToken', {})
                symbol = base.get('symbol')
                addr = item.get('pairAddress') or base.get('address')
                if addr:
                    db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                    return symbol, addr
    except Exception as e:
        logger.error(f"Error in first DEXScreener search for {s}: {e}")
    
    # Try another approach for unknown tokens - search using Jupiter API
    try:
        logger.info(f"Trying Jupiter API for token: {s}")
        jupiter_url = "https://token.jup.ag/strict"
        resp = requests.get(jupiter_url, timeout=10)
        resp.raise_for_status()
        
        token_list = resp.json()
        # Look for exact match first
        for token in token_list:
            if token.get('symbol', '').upper() == s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
        
        # Try partial match if no exact match
        for token in token_list:
            if s in token.get('symbol', '').upper() or token.get('symbol', '').upper() in s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
    except Exception as e:
        logger.error(f"Error in Jupiter search for {s}: {e}")
        
    # Try one more approach - check Birdeye API
    try:
        logger.info(f"Trying Birdeye API for token: {s}")
        headers = {
            "X-API-KEY": os.getenv("BIRDEYE_API_KEY", ""),  # Optional API key
            "User-Agent": "Mozilla/5.0"
        }
        birdeye_url = f"https://public-api.birdeye.so/public/tokenlist/solana?search={s}"
        resp = requests.get(birdeye_url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        token_list = resp.json().get("data", [])
        # Look for exact match first
        for token in token_list:
            if token.get('symbol', '').upper() == s:
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
                        
        # Try partial match
        for token in token_list:
            if s in token.get('symbol', '').upper():
                symbol = token.get('symbol')
                addr = token.get('address')
                if addr:
                    # Verify with DexScreener
                    dex_data = fetch_data(addr)
                    if dex_data:
                        db.setex(cache_key, 86400, json.dumps({"symbol": symbol, "address": addr}))
                        return symbol, addr
    except Exception as e:
        logger.error(f"Error in Birdeye search for {s}: {e}")
    
    return None, None

def format_token_data(d: dict) -> str:
    """Format token data into a Twitter-friendly message"""
    if not d:
        return "Could not fetch token data at this time."
        
    lines = [
        f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
        f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
        f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
        d['link']
    ]
    return "\n".join(lines)

def extract_ticker(text: str) -> str:
    """Extract ticker symbol from text"""
    matches = TICKER_RE.findall(text)
    return matches[0] if matches else None

async def handle_mention(ev: dict):
    events = ev.get('tweet_create_events') or []
    if not events or not isinstance(events, list) or not events[0].get("text"):
        logger.warning("Skipping invalid or empty mention event")
        return {"message": "no valid mention"}

    tweet_text = events[0]['text'].replace('@askdegen', '').strip()
    tweet_id = events[0]['id_str']
    
    # Extract ticker from the tweet text
    ticker_from_regex = extract_ticker(tweet_text)
    
    # Find token either explicitly mentioned or in question
    explicit_token = next((w for w in tweet_text.split() if w.startswith('$') or ADDR_RE.match(w)), None)
    token = explicit_token or (f"${ticker_from_regex}" if ticker_from_regex else None)
    
    reply = None
    token_data = None

    if token:
        sym, addr = resolve_token(token)
        logger.info(f"Resolved {token} to symbol={sym}, address={addr}")
        
        if addr:
            token_data = fetch_data(addr)
            
            # If only token is mentioned, return just the data
            if tweet_text.strip() == token or len(tweet_text.split()) <= 2:
                reply = format_token_data(token_data)
            # Otherwise, we have a question about the token - use Perplexity with token data
            else:
                reply = ask_perplexity(tweet_text, token_data, max_tokens=160)
                if reply:
                    # Combine token data and Perplexity response
                    token_text = format_token_data(token_data)
                    reply = f"{token_text}\n\n{reply}"
                else:
                    # Fall back to Grok with token data
                    prompt = f"You are a professional crypto market analyst. Given: {json.dumps(token_data)}, reply to an investor in <140 characters about their query: {tweet_text}"
                    analysis = ask_grok(prompt, tweet_text, max_tokens=160)
                    reply = f"{format_token_data(token_data)}\n\n{analysis}"
        elif ticker_from_regex:
            # Try Perplexity for general information about the token
            query = f"What's the latest on ${ticker_from_regex} crypto token? Latest price, trends, and news in brief."
            reply = ask_perplexity(query, max_tokens=180)
        
    # If no token data and no reply yet, use Perplexity for general crypto questions
    if not reply and any(word in tweet_text.lower() for word in ["crypto", "token", "coin", "market", "price", "sol", "btc", "eth"]):
        reply = ask_perplexity(tweet_text, max_tokens=180)
        
    # Final fallback to Grok for any other questions
    if not reply:
        system_msg = "Professional crypto analyst: provide concise, insightful response under 240 characters."
        reply = ask_grok(system_msg, tweet_text, max_tokens=180)

    # Ensure tweet isn't too long
    tweet = reply.strip()
    if len(tweet) > 280:
        tweet = tweet[:280]
        if '.' in tweet:
            tweet = tweet.rsplit('.', 1)[0] + '.'
        else:
            tweet = tweet.rsplit(' ', 1)[0] + '...'
            
    # Post reply
    try:
        x_client.create_tweet(text=tweet, in_reply_to_tweet_id=int(tweet_id))
        logger.info(f"Posted reply to {tweet_id}")
    except Exception as e:
        logger.error(f"Error posting reply: {e}")
        try:
            x_api.update_status(status=tweet, in_reply_to_status_id=tweet_id)
            logger.info(f"Posted reply via API to {tweet_id}")
        except Exception as e2:
            logger.error(f"Second error posting reply: {e2}")
            
    return {'message': 'ok'}

async def degen_hourly_loop():
    while True:
        try:
            d = fetch_data(DEGEN_ADDR)
            if not d:
                logger.error("Failed to fetch DEGEN data for hourly update")
                await asyncio.sleep(600)  # Try again in 10 minutes
                continue
                
            # Generate the stats card
            card = [
                f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                f"1h {'🟢' if d['change_1h'] >= 0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h'] >= 0 else '🔴'}{d['change_24h']:+.2f}%",
                d['link']
            ]
            
            # Try Perplexity first for latest news/insights
            perplexity_prompt = f"Latest news and market insights for DEGEN token on Solana. Current price: ${d['price_usd']:.6f}, 24h change: {d['change_24h']:+.2f}%. Keep it brief and actionable."
            analysis = ask_perplexity(perplexity_prompt, d, max_tokens=120)  # Reduced token count
            
            # Fall back to Grok if needed
            if not analysis:
                sys_msg = "You are a DEGEN community insider. Write a 2-sentence hourly update based on this data. Be enthusiastic but grounded. Do not mention Solana."
                analysis = ask_grok(sys_msg, json.dumps(d), max_tokens=120)  # Reduced token count
            
            # Calculate available characters for analysis
            # Fixed card size + 2 extra newlines
            card_size = sum(len(line) for line in card) + 3  # +3 for newlines
            max_analysis_chars = 275 - card_size  # 275 instead of 280 for safety margin
            
            # Truncate analysis if needed, ensuring we end on a complete sentence
            if len(analysis) > max_analysis_chars:
                analysis = analysis[:max_analysis_chars]
                # Find the last sentence boundary
                last_period = analysis.rfind('.')
                if last_period > max_analysis_chars * 0.6:  # Only truncate if we're not losing too much
                    analysis = analysis[:last_period+1]
                else:
                    # If no good sentence boundary, find last space
                    last_space = analysis.rfind(' ')
                    analysis = analysis[:last_space] + '...'
                
            # Combine card and analysis with proper spacing
            tweet = "\n".join(card + ["", analysis])
                    
            try:
                x_client.create_tweet(text=tweet)
                logger.info("Hourly promo posted")
            except Exception as e:
                logger.error(f"Error posting via client: {e}")
                try:
                    x_api.update_status(tweet)
                    logger.info("Hourly promo posted via API")
                except Exception as e2:
                    logger.error(f"Second error posting hourly update: {e2}")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
            
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        try:
            last = db.get(f"{REDIS_PREFIX}last_tweet_id")
            since_id = int(last) if last else None
            res = x_client.get_users_mentions(
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
                    ev = {'tweet_create_events': [{'id_str': str(tw.id), 'text': tw.text, 'user': {'screen_name': users.get(tw.author_id, '?')}}]}
                    try:
                        await handle_mention(ev)
                    except Exception as e:
                        logger.error(f"Mention error: {e}")
                    db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                    db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
            
            # Adjust polling frequency based on recent activity
            lm = db.get(f"{REDIS_PREFIX}last_mention")
            if lm and time.time() - int(lm) < 3600:
                # More frequent checks if there was activity in the last hour
                await asyncio.sleep(60)
            else:
                # Otherwise check less frequently
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Poll loop error: {e}")
            await asyncio.sleep(300)  # Back off on error

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root():
    return {'message': 'Degen Meme Bot is live.'}

@app.post('/test')
async def test_bot(r: Request):
    data = await r.json()
    ev = {'tweet_create_events': [{'id_str': '0', 'text': data.get('text', ''), 'user': {'screen_name': 'test'}}]}
    return await handle_mention(ev)