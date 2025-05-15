from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, Response
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import json
from time import sleep
import hmac
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", 
    "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", 
    "X_WEBHOOK_ENV", "X_WEBHOOK_SECRET"  # Add these two new required variables
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Tweepy client (@askdegen, Basic $200/month plan)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Also create an API v1.1 client for account activity API setup
auth = tweepy.OAuth1UserHandler(
    os.getenv("X_API_KEY"),
    os.getenv("X_API_SECRET"),
    os.getenv("X_ACCESS_TOKEN"),
    os.getenv("X_ACCESS_TOKEN_SECRET")
)
api_v1 = tweepy.API(auth)

logger.info(f"Authenticated as: {x_client.get_me().data.username}")

# Redis client configuration remains the same
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
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"
REDIS_CACHE_PREFIX = "degen:"
DEGEN_ADDRESS = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
X_WEBHOOK_ENV = os.getenv("X_WEBHOOK_ENV")  # The environment label for the webhook (e.g., "production")
X_WEBHOOK_SECRET = os.getenv("X_WEBHOOK_SECRET")  # Secret for validating webhook requests

# Your existing functions remain the same
def fetch_dexscreener_data(address: str, retries=3, backoff=2) -> dict:
    """Fetch token metrics from DexScreener with caching."""
    cache_key = f"{REDIS_CACHE_PREFIX}dex:{address}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis get error: {e}")

    for attempt in range(retries):
        try:
            r = requests.get(f"{DEXSCREENER_URL}{address}", timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or not data[0]:
                return {}
            pair = data[0]
            result = {
                "token_symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
                "price_usd": float(pair.get("priceUsd", 0)),
                "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                "volume_usd": float(pair.get("volume", {}).get("h24", 0)),
                "market_cap": float(pair.get("marketCap", 0))
            }
            redis_client.setex(cache_key, 300, json.dumps(result))
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                sleep(backoff)
                backoff *= 2
                continue
            return {}
        except requests.RequestException as e:
            logger.error(f"DexScreener error: {e}")
            return {}
    return {}

def resolve_token(query: str) -> tuple:
    """Resolve query to $TOKEN and address via X sentiment."""
    # Existing function code...
    query = query.strip().lower()
    is_contract = re.match(r"^[A-Za-z0-9]{43,44}$", query)
    is_degen = query in ["degen", "$degen"] or (is_contract and query == DEGEN_ADDRESS)

    if is_degen:
        data = fetch_dexscreener_data(DEGEN_ADDRESS)
        return "DEGEN", DEGEN_ADDRESS, data
    elif is_contract:
        data = fetch_dexscreener_data(query)
        if data.get("token_symbol", "Unknown") != "Unknown":
            return data["token_symbol"].upper(), query, data
        system = "Crypto analyst. Find $TOKEN ticker for Solana address from X. JSON: {'token': str, 'address': str}"
        user_msg = f"Contract: {query}. Find ticker from X."
    else:
        token = query.replace("$", "").upper()
        system = "Crypto analyst. Find Solana address for $TOKEN from X. JSON: {'token': str, 'address': str}"
        user_msg = f"Ticker: {token}. Find address from X."

    cache_key = f"{REDIS_CACHE_PREFIX}resolve:{query}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            token = data.get("token", "UNKNOWN").upper()
            address = data.get("address", "")
            return token, address, fetch_dexscreener_data(address) if address else {}
    except redis.RedisError:
        pass

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 100, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        token = data.get("token", "UNKNOWN").upper()
        address = data.get("address", "")
        redis_client.setex(cache_key, 3600, json.dumps({"token": token, "address": address}))
        return token, address, fetch_dexscreener_data(address) if address else {}
    except Exception as e:
        logger.error(f"Resolve error: {e}")
        return "UNKNOWN", "", {}

def handle_confession(confession: str, user: str, tid: str) -> str:
    """Parse and tweet a Degen Confession."""
    # Existing function code...
    system = "Witty crypto bot. Summarize confession into a fun, anonymized tweet with a challenge. ≤750 chars, use only what's needed. JSON: {'tweet': str}"
    user_msg = f"Confession: {confession}. Hype degen spirit, add challenge, keep it short."
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 750, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        tweet = data.get("tweet", "Degen spilled a wild tale! Share yours! #DegenConfession")[:750]
        tweet_response = x_client.create_tweet(text=tweet)
        link = f"https://x.com/askdegen/status/{tweet_response.data['id']}"
        return f"Your confession's live! See: {link}"
    except Exception as e:
        logger.error(f"Confession error: {e}")
        return "Confession failed. Try again!"

def analyze_hype(query: str, token: str, address: str, dexscreener_data: dict, tid: str) -> str:
    """Analyze hype for a coin with conversation memory."""
    # Existing function code...
    context_key = f"{REDIS_CACHE_PREFIX}context:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError:
        prior_context = {"query": "", "response": ""}

    is_degen = token == "DEGEN" or address == DEGEN_ADDRESS
    system = (
        "Witty crypto analyst. Analyze coin hype from X and market data. For $DEGEN, stay positive, compare to $DOGE/$SHIB's ups/downs. "
        "Reply ≤150 chars, 1-2 sentences. JSON: {'reply': str, 'hype_score': int}"
    )
    user_msg = (
        f"Coin: {token}. Market: {json.dumps(dexscreener_data)}. Prior: Query: {prior_context['query']}, Reply: {prior_context['response']}. "
        f"Fun, short reply, hype score. {'Stay positive for $DEGEN.' if is_degen else ''}"
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 150, "temperature": 0.7}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        reply = data.get("reply", "No vibe on X. Try $BONK!")[:150]
        redis_client.setex(context_key, 86400, json.dumps({"query": query, "response": reply}))
        return reply
    except Exception as e:
        logger.error(f"Hype error: {e}")
        return "No vibe on X. Try $BONK!"

# Verify the X signature for incoming webhook events
def verify_x_signature(request_body: bytes, x_signature: str) -> bool:
    """Verify the X signature for webhook events."""
    if not x_signature:
        return False
    
    # Create a signature using your webhook secret
    expected_signature = hmac.new(
        X_WEBHOOK_SECRET.encode('utf-8'),
        msg=request_body,
        digestmod=hashlib.sha256
    ).digest()
    
    # Compare signatures
    try:
        received_signature = base64.b64decode(x_signature.split('=')[1])
        return hmac.compare_digest(expected_signature, received_signature)
    except:
        return False

# Add these new endpoints for X Account Activity API setup

@app.get("/webhook")
async def webhook_challenge(request: Request, crc_token: str = None):
    """Handle the CRC (Challenge-Response Check) from X API."""
    if not crc_token:
        return JSONResponse(status_code=400, content={"message": "Missing crc_token"})
    
    # Create the response
    sha256_hash_digest = hmac.new(
        X_WEBHOOK_SECRET.encode('utf-8'),
        msg=crc_token.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    
    # Return the response
    response = {
        'response_token': f'sha256={base64.b64encode(sha256_hash_digest).decode("utf-8")}'
    }
    return JSONResponse(content=response)

@app.post("/webhook")
async def webhook_event(request: Request, x_twitter_webhooks_signature: str = Header(None)):
    """Handle incoming webhook events from X."""
    # Get the raw request body
    body = await request.body()
    
    # Verify the request is from X
    if not verify_x_signature(body, x_twitter_webhooks_signature):
        logger.warning("Invalid X signature")
        return Response(status_code=401)
    
    # Parse the request data
    try:
        data = json.loads(body.decode('utf-8'))
        logger.info(f"Received webhook event: {json.dumps(data)[:200]}...")
        
        # Check if this is a tweet_create_events
        if "tweet_create_events" in data:
            # Forward to your existing handler
            return await handle_mention(data)
        
        # Acknowledge other event types
        return JSONResponse({"message": "Event received"}, status_code=200)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in webhook payload")
        return Response(status_code=400)
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return Response(status_code=500)

@app.post("/")
async def handle_mention(data: dict):
    """Handle @askdegen mentions and comments."""
    try:
        # Ensure we have tweet_create_events
        if "tweet_create_events" not in data or not data["tweet_create_events"]:
            logger.warning("No tweet_create_events in payload")
            return JSONResponse({"message": "No tweet events"}, status_code=400)
        
        evt = data["tweet_create_events"][0]
        txt = evt.get("text", "").replace("@askdegen", "").strip()
        user = evt.get("user", {}).get("screen_name", "")
        tid = evt.get("id_str", "")
        reply_tid = evt.get("in_reply_to_status_id_str", tid) or tid

        if not all([txt, user, tid]):
            logger.warning(f"Invalid tweet data: {txt=}, {user=}, {tid=}")
            return JSONResponse({"message": "Invalid tweet"}, status_code=400)

        logger.info(f"Processing: {tid}, {user}, {txt}")

        if txt.lower().startswith("degen confession:"):
            reply = handle_confession(txt[16:].strip(), user, tid)
        else:
            query = next((w[1:] for w in txt.split() if w.startswith("$") and len(w) > 1), None) or \
                    next((w for w in txt.split() if re.match(r"^[A-Za-z0-9]{43,44}$", w)), None) or \
                    "most hyped coin"
            token, address, data = resolve_token(query)
            reply = analyze_hype(query, token, address, data, tid)

        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid))
            logger.info(f"Replied: {reply} to tweet {reply_tid}")
        except tweepy.errors.Forbidden:
            x_client.create_tweet(text=reply)
            logger.info(f"Created new tweet: {reply} (couldn't reply)")
        
        return JSONResponse({"message": "Success"}, status_code=200)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New function to register the webhook on application startup
@app.on_event("startup")
async def register_webhook():
    """Register the webhook with X's Account Activity API on startup."""
    try:
        # Define your webhook URL (must be HTTPS)
        webhook_url = os.getenv("WEBHOOK_URL")  # Add this to your .env file
        
        if not webhook_url:
            logger.error("Missing WEBHOOK_URL environment variable")
            return
        
        logger.info(f"Registering webhook {webhook_url} with X...")
        
        # First, check if there's already a webhook registered
        try:
            webhooks = api_v1.get_webhooks(X_WEBHOOK_ENV)
            if webhooks:
                for webhook in webhooks:
                    logger.info(f"Found existing webhook: {webhook.url}")
                    if webhook.url == webhook_url:
                        logger.info("Webhook already registered")
                        
                        # Check subscriptions
                        subscriptions = api_v1.get_webhook_subscriptions(X_WEBHOOK_ENV, webhook.id)
                        if not subscriptions:
                            logger.info("No active subscription, subscribing...")
                            api_v1.subscribe_to_webhook(X_WEBHOOK_ENV, webhook.id)
                            logger.info("Subscribed to webhook events")
                        else:
                            logger.info("Subscription already active")
                        
                        return
                    
                    # If there's a different webhook, delete it
                    logger.info(f"Deleting old webhook: {webhook.url}")
                    api_v1.delete_webhook(X_WEBHOOK_ENV, webhook.id)
        except Exception as e:
            logger.error(f"Error checking existing webhooks: {e}")
        
        # Register new webhook
        response = api_v1.register_webhook(X_WEBHOOK_ENV, webhook_url)
        webhook_id = response.id
        logger.info(f"Registered new webhook: {webhook_id}")
        
        # Subscribe to webhook events
        api_v1.subscribe_to_webhook(X_WEBHOOK_ENV, webhook_id)
        logger.info("Subscribed to webhook events")
        
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")

# Add a route to manually trigger the webhook registration
@app.get("/register-webhook")
@app.post("/register-webhook")
async def trigger_webhook_registration():
    """Manually trigger webhook registration."""
    try:
        await register_webhook()
        return JSONResponse({"message": "Webhook registration attempted"}, status_code=200)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a testing endpoint for manually triggering the bot
@app.post("/test")
async def test_bot(request: Request):
    """Test the bot with a simulated mention."""
    try:
        body = await request.json()
        text = body.get("text", "@askdegen Tell me about $DEGEN")
        user = body.get("user", "test_user")
        
        # Create a simulated tweet event
        test_event = {
            "tweet_create_events": [
                {
                    "id_str": "123456789",
                    "text": text,
                    "user": {
                        "screen_name": user
                    }
                }
            ]
        }
        
        # Process the simulated event
        response = await handle_mention(test_event)
        return response
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
