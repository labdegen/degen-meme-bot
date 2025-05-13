# api/webhook.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re

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
    """Use Grok to analyze X data for token sentiment, mentions, and momentum."""
    logger.info(f"Fetching X data for query: {query}")
    system = (
        "You’re a crypto degen analyzing X posts. Analyze recent posts (as of May 13, 2025) about the token. "
        "Provide: tweet count, momentum (high/low based on frequency), sentiment (bullish/bearish/neutral), "
        "and big accounts (>10,000 followers) mentioning it. Return in JSON format."
    )
    user_msg = (
        f"Analyze X posts for {query} (token or address). Summarize: "
        "1. Approx. number of tweets (e.g., dozens, hundreds). "
        "2. Momentum (high: frequent posts; low: sparse). "
        "3. Sentiment (bullish/bearish/neutral, with % if possible). "
        "4. Big accounts (>10,000 followers) mentioning it (handles, follower counts). "
        "Return JSON: {{'tweets': str, 'momentum': str, 'sentiment': str, 'big_accounts': list of {{'handle': str, 'followers': int}}}}"
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
    """Call Grok for a ≤240-char reply based on token data or conversational input."""
    system = (
        "You’re a degenerate crypto gambler, snarky but useful. Keep it ≤240 chars. "
        "For tokens, use X data (tweets, momentum, sentiment, big accounts) to mock pumps/rugs or hype. "
        "For general text, reply conversationally or analyze sentiment."
    )
    if not token_data.get("no_data"):
        user_msg = (
            f"Token: {query} ({token_data['symbol']}). "
            f"Tweets: {token_data['tweets']}. Momentum: {token_data['momentum']}. "
            f"Sentiment: {token_data['sentiment']}. "
            f"Big accounts: {', '.join([acc['handle'] for acc in token_data['big_accounts']] or ['None'])}. "
            "Give a punchy degen take."
        )
    else:
        user_msg = (
            f"Tweet: {tweet_text}. No token data found. "
            "Reply conversationally as a crypto degen or analyze X sentiment."
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
        "temperature": 0.9
    }
    logger.info(f"Grok request: headers={{'Authorization': 'Bearer [REDACTED]', 'Content-Type': 'application/json'}}, body={body}")
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        if r.status_code == 400:
            logger.error(f"Grok API 400 error: {r.text}")
            return "Yo, my circuits are fried! Try another token or vibe check."
        r.raise_for_status()
        response = r.json()
        logger.info(f"Grok response: {response}")
        text = response["choices"][0]["message"]["content"].strip()
        return text[:240]
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        return "Yo, my circuits are fried! Try another token or vibe check."

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