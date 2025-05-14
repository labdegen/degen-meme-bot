# api/webhook.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "GROK_API_KEY", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing environment variable: {var}")
        raise RuntimeError(f"Missing environment variable: {var}")

# Tweepy client (@askdegen creds)
try:
    x_client = tweepy.Client(
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    user = x_client.get_me()
    logger.info(f"X API authenticated as: {user.data.username}")
except tweepy.TweepyException as e:
    logger.error(f"X API authentication failed: {str(e)}")
    raise RuntimeError(f"X API authentication failed: {str(e)}")

# Redis client
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info("Redis connection successful")
except redis.RedisError as e:
    logger.error(f"Redis connection failed: {str(e)}")
    raise RuntimeError(f"Redis connection failed: {str(e)}")

# Grok config
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Token regex for $TOKEN or contract address
HEX_REGEX = re.compile(r"^(0x[a-fA-F0-9]{40}|[A-Za-z0-9]{43,44})$")

def fetch_token_data(query: str, tid: str) -> dict:
    """Use Grok to analyze X data for token sentiment, mentions, and momentum from the last 7 days."""
    logger.info(f"Fetching X data for query: {query}, tid: {tid}")
    # Check Redis for prior context
    context_key = f"conversation:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError as e:
        logger.error(f"Redis get failed: {str(e)}")
        prior_context = {"query": "", "response": ""}

    # Calculate date range (past 7 days)
    today = datetime.utcnow()
    start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    system = (
        "You are a cynical crypto analyst. Analyze X posts from the last 7 days for the given token. "
        "Focus on tweet volume, momentum (high: >100 posts/day; medium: 10-100; low: <10), sentiment (bullish/bearish/neutral, with %), "
        "and influential accounts (>10,000 followers, active in last 30 days). Return JSON with concise metrics. "
        "Only include verified, active accounts to avoid outdated or fabricated data."
    )
    user_msg = (
        f"Analyze X posts for {query} (token or address) from {start_date} to {end_date}. "
        f"Prior context: Query: {prior_context['query']}, Response: {prior_context['response']}. "
        "Provide: "
        "1. Tweet count (e.g., '50-100'). "
        "2. Momentum (high/medium/low). "
        "3. Sentiment (bullish/bearish/neutral, with %). "
        "4. Influential accounts (>10,000 followers, active in last 30 days, with handles and follower counts). "
        "Return JSON: {'tweets': str, 'momentum': str, 'sentiment': str, 'big_accounts': [{'handle': str, 'followers': int}]}"
    )
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    logger.info(f"Grok request: headers={{'Authorization': 'Bearer [REDACTED]', 'Content-Type': 'application/json'}}, body={body}")
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        if r.status_code == 400:
            logger.error(f"Grok API 400 error: {r.text}")
            return {"no_data": True}
        r.raise_for_status()
        response = r.json()
        logger.info(f"Grok response: {response}")
        text = response["choices"][0]["message"]["content"].strip()
        try:
            data = json.loads(text)
            return {
                "no_data": False,
                "tweets": data.get("tweets", "Unknown"),
                "momentum": data.get("momentum", "Unknown"),
                "sentiment": data.get("sentiment", "Unknown"),
                "big_accounts": data.get("big_accounts", []),
                "symbol": query.replace("$", "").upper()
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Grok response as JSON: {text}")
            return {"no_data": True}
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        return {"no_data": True}

def generate_reply(token_data: dict, query: str, tweet_text: str, tid: str) -> str:
    """Generate a natural, conversational reply with context retention."""
    logger.info(f"Generating reply for query: {query}, tid: {tid}")
    # Check Redis for prior context
    context_key = f"conversation:{tid}"
    try:
        context = redis_client.get(context_key)
        prior_context = json.loads(context) if context else {"query": "", "response": ""}
    except redis.RedisError as e:
        logger.error(f"Redis get failed: {str(e)}")
        prior_context = {"query": "", "response": ""}

    system = (
        "You are a cynical crypto trader. Respond in a dry, conversational tone under 280 characters. "
        "React naturally like a seasoned trader, don’t repeat or summarize the query. For tokens, use X data to call out trends, risks, and key accounts with a skeptical edge. "
        "For general text, stay sharp and analytical. Use prior context to stay on-topic, cite only verified, active accounts."
    )
    if not token_data.get("no_data"):
        accounts = ", ".join([f"@{acc['handle']}" for acc in token_data["big_accounts"]]) or "no big names"
        user_msg = (
            f"Token: {query} ({token_data['symbol']}). "
            f"Tweets: {token_data['tweets']}. Momentum: {token_data['momentum']}. "
            f"Sentiment: {token_data['sentiment']}. Accounts: {accounts}. "
            f"Prior: Query: {prior_context['query']}, Response: {prior_context['response']}. "
            "Give a concise, actionable take like a trader, max 280 chars."
        )
    else:
        user_msg = (
            f"Query: {tweet_text}. No token data found. "
            f"Prior: Query: {prior_context['query']}, Response: {prior_context['response']}. "
            "Engage conversationally with a sharp, analytical edge, max 280 chars."
        )
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 280,
        "temperature": 0.7
    }
    logger.info(f"Grok request: headers={{'Authorization': 'Bearer [REDACTED]', 'Content-Type': 'application/json'}}, body={body}")
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
        if r.status_code == 400:
            logger.error(f"Grok API 400 error: {r.text}")
            return "No real buzz on X for this. Try a token with more noise."
        r.raise_for_status()
        response = r.json()
        logger.info(f"Grok response: {response}")
        text = response["choices"][0]["message"]["content"].strip()

        # Store context in Redis
        try:
            redis_client.setex(context_key, 3600, json.dumps({"query": query or tweet_text, "response": text}))
        except redis.RedisError as e:
            logger.error(f"Redis set failed: {str(e)}")

        return text[:280]
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        return "No real buzz on X for this. Try a token with more noise."

@app.post("/")
async def handle_mention(data: dict):
    """Handle X test mentions for @askdegen (free tier)."""
    logger.info(f"Received payload: {json.dumps(data, indent=2)}")
    try:
        if "tweet_create_events" not in data or not data["tweet_create_events"]:
            logger.error(f"Invalid payload: {data}")
            return JSONResponse({"message": "No tweet data"}, status_code=400)

        evt = data["tweet_create_events"][0]
        txt = evt.get("text", "")
        user = evt.get("user", {}).get("screen_name", "")
        tid = evt.get("id_str", "")
        in_reply_to_status_id = evt.get("in_reply_to_status_id_str", None)

        if not all([txt, user, tid]):
            logger.error(f"Missing tweet data: text={txt}, user={user}, tid={tid}")
            return JSONResponse({"message": "Invalid tweet data"}, status_code=400)

        logger.info(f"Processing tweet ID: {tid}, user: {user}, text: {txt}, in_reply_to: {in_reply_to_status_id}")

        # Use in_reply_to_status_id if available (for threaded conversations)
        reply_tid = in_reply_to_status_id if in_reply_to_status_id and in_reply_to_status_id.isdigit() else tid

        # Validate reply_tid
        if not reply_tid.isdigit() or len(reply_tid) < 15:
            logger.error(f"Invalid reply tweet ID: {reply_tid}")
            # Generate reply for standalone post
            words = txt.split()
            tok = None
            for w in words:
                if w.startswith("$") and len(w) > 1:
                    tok = w[1:]
                    break
                if HEX_REGEX.match(w):
                    tok = w
                    break

            try:
                if tok:
                    token_data = fetch_token_data(tok, tid)
                    reply_content = generate_reply(token_data, tok, txt, tid)
                else:
                    token_data = {"no_data": True}
                    reply_content = generate_reply(token_data, "", txt, tid)

                tweet_response = x_client.create_tweet(text=reply_content)
                logger.info(f"Standalone reply posted: {reply_content}, response: {tweet_response}")
                return JSONResponse({"message": "Replied to mention (standalone)"}, status_code=200)
            except tweepy.TweepyException as e:
                logger.error(f"Failed to post standalone reply: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

        # Check for $TOKEN or contract address
        words = txt.split()
        tok = None
        for w in words:
            if w.startswith("$") and len(w) > 1:
                tok = w[1:]
                break
            if HEX_REGEX.match(w):
                tok = w
                break

        try:
            if tok:
                # Token or address mentioned
                token_data = fetch_token_data(tok, tid)
                reply_content = generate_reply(token_data, tok, txt, tid)
            else:
                # No token/address, conversational reply
                token_data = {"no_data": True}
                reply_content = generate_reply(token_data, "", txt, tid)

            reply = reply_content
            max_reply_length = 280
            reply = reply[:max_reply_length]

            try:
                tweet_response = x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(reply_tid))
                logger.info(f"Replied to tweet {reply_tid} with: {reply}, response: {tweet_response}")
            except tweepy.errors.Forbidden as e:
                logger.warning(f"Threaded reply failed for tweet {reply_tid}: {str(e)}; posting standalone")
                tweet_response = x_client.create_tweet(text=reply)
                logger.info(f"Standalone reply posted: {reply}, response: {tweet_response}")
            except tweepy.TweepyException as e:
                logger.error(f"Failed to post reply to tweet {reply_tid}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

            return JSONResponse({"message": "Replied to mention"}, status_code=200)
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"handle_mention error: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=500)