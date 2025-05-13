# api/webhook.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment variables
required_env_vars = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "GROK_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing environment variable: {var}")
        raise RuntimeError(f"Missing environment variable: {var}")

# Tweepy client (@askdegen creds)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Grok config
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Token regex for $TOKEN or contract address
HEX_REGEX = re.compile(r"^(0x[a-fA-F0-9]{40}|[A-Za-z0-9]{43,44})$")  # Ethereum or Solana addresses

def fetch_token_data(query: str) -> dict:
    """Use Grok to analyze X data for token sentiment, mentions, and momentum from the last 7 days."""
    logger.info(f"Fetching X data for query: {query}")
    # Calculate date range (past 7 days)
    today = datetime.utcnow()
    start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    system = (
        "You are an analytical crypto researcher. Analyze X posts from the last 7 days for the given token. "
        "Focus on tweet volume, momentum (high/low based on post frequency), sentiment (bullish/bearish/neutral, with %), "
        "and influential accounts (>10,000 followers, active in the last 30 days). Return JSON with precise metrics."
    )
    user_msg = (
        f"Analyze X posts for {query} (token or address) from {start_date} to {end_date}. Provide: "
        "1. Tweet count (e.g., '50-100'). "
        "2. Momentum (high: >100 posts/day; medium: 10-100; low: <10). "
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
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        if r.status_code == 400:
            logger.error(f"Grok API 400 error: {r.text}")
            return {"no_data": True}
        r.raise_for_status()
        response = r.json()
        logger.info(f"Grok response: {response}")
        text = response["choices"][0]["message"]["content"].strip()
        import json
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

def generate_reply(token_data: dict, query: str, tweet_text: str) -> str:
    """Generate a dry, analytical reply based on token data or conversational input."""
    system = (
        "You are an expert crypto analyst. Deliver dry, data-driven responses under 240 characters. "
        "For tokens, use X data (tweets, momentum, sentiment, accounts) to assess trends. Highlight risks or manipulation. "
        "For general text, provide analytical insight or sentiment analysis. Avoid humor."
    )
    if not token_data.get("no_data"):
        accounts = ", ".join([f"@{acc['handle']} ({acc['followers']:,})" for acc in token_data["big_accounts"]]) or "None"
        user_msg = (
            f"Token: {query} ({token_data['symbol']}). "
            f"Tweets: {token_data['tweets']}. Momentum: {token_data['momentum']}. "
            f"Sentiment: {token_data['sentiment']}. Accounts: {accounts}. "
            "Assess trends and risks in a concise, analytical manner."
        )
    else:
        user_msg = (
            f"Tweet: {tweet_text}. No token data found. "
            "Provide analytical insight or analyze X sentiment for the query."
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
        "max_tokens": 200,
        "temperature": 0.7
    }
    logger.info(f"Grok request: headers={{'Authorization': 'Bearer [REDACTED]', 'Content-Type': 'application/json'}}, body={body}")
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        if r.status_code == 400:
            logger.error(f"Grok API 400 error: {r.text}")
            return "Insufficient X data. Query a more active token or topic."
        r.raise_for_status()
        response = r.json()
        logger.info(f"Grok response: {response}")
        text = response["choices"][0]["message"]["content"].strip()
        return text[:240]
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        return "Insufficient X data. Query a more active token or topic."

@app.post("/")
async def handle_mention(data: dict):
    """Handle Twitter webhook for @askdegen mentions."""
    logger.info(f"Received payload: {data}")
    if "tweet_create_events" not in data or not data["tweet_create_events"]:
        logger.error(f"Invalid webhook payload: {data}")
        return JSONResponse({"message": "No tweet data"}, status_code=400)

    evt = data["tweet_create_events"][0]
    txt = evt.get("text", "")
    user = evt.get("user", {}).get("screen_name", "")
    tid = evt.get("id_str", "")

    if not all([txt, user, tid]):
        logger.error(f"Missing tweet data: text={txt}, user={user}, tid={tid}")
        return JSONResponse({"message": "Invalid tweet data"}, status_code=400)

    logger.info(f"Processing tweet ID: {tid}, user: {user}, text: {txt}")

    # Validate tid early
    if not tid.isdigit() or len(tid) < 15:
        logger.error(f"Invalid tweet ID: {tid}")
        # Generate reply anyway for standalone post
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
                token_data = fetch_token_data(tok)
                reply_content = generate_reply(token_data, tok, txt)
            else:
                token_data = {"no_data": True}
                reply_content = generate_reply(token_data, "", txt)

            reply = f"Re: {txt[:50]}... {reply_content}"[:280]
            tweet_response = x_client.create_tweet(text=reply)
            logger.info(f"Standalone reply posted due to invalid tid: {reply}, response: {tweet_response}")
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
            token_data = fetch_token_data(tok)
            reply_content = generate_reply(token_data, tok, txt)
        else:
            # No token/address, conversational reply
            token_data = {"no_data": True}
            reply_content = generate_reply(token_data, "", txt)

        reply = reply_content  # No @user tag
        max_reply_length = 280 - 1
        reply = reply[:max_reply_length]

        try:
            tweet_response = x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
            logger.info(f"Replied to tweet {tid} with: {reply}, response: {tweet_response}")
        except tweepy.errors.Forbidden as e:
            logger.warning(f"Threaded reply failed for tweet {tid}: {str(e)}; posting standalone")
            reply = f"Re: {txt[:50]}... {reply}"[:280]
            tweet_response = x_client.create_tweet(text=reply)
            logger.info(f"Standalone reply posted: {reply}, response: {tweet_response}")
        except tweepy.TweepyException as e:
            logger.error(f"Failed to post reply to tweet {tid}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

        return JSONResponse({"message": "Replied to mention"}, status_code=200)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"handle_mention error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)