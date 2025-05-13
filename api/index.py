from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize X API client (using @askdegen creds)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Grok endpoint & key
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Stubbed symbol→address map
SYMBOL_ADDRESS_MAP = {
    "DEGEN": "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
}

def fetch_token_data(symbol: str):
    """
    TEST STUB: replace with your real Codex.io or CoinGecko fetch.
    """
    address = SYMBOL_ADDRESS_MAP.get(symbol, symbol)
    logger.info(f"Using address {address} for symbol {symbol}")
    return {
        "address": address,
        "name": f"{symbol} Token",
        "symbol": symbol,
        "totalSupply": "1,000,000,000",
        "circulatingSupply": "750,000,000"
    }

def generate_reply(token_data: dict, token_name: str, user_handle: str) -> str:
    """
    Ask Grok for a fresh, degen-style analysis that fits in 240 chars.
    """
    system = (
        "You are a degenerate crypto gambler. Keep it concise (<=240 chars), "
        "snarky but useful, mocking moonbois and calling out FUD. "
        "Always include a quick take and one data point."
    )
    user_msg = (
        f"User @{user_handle} mentioned ${token_name}. "
        f"Token info: name={token_data['name']}, "
        f"symbol={token_data['symbol']}, "
        f"totalSupply={token_data['totalSupply']}, "
        f"circulatingSupply={token_data['circulatingSupply']}. "
        "Give me a punchy degen analysis."
    )

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 200,
        "temperature": 0.9
    }

    resp = requests.post(GROK_URL, json=body, headers=headers)
    if resp.status_code != 200:
        logger.error(f"Grok error: {resp.text}")
        raise HTTPException(status_code=500, detail="Failed to generate analysis")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Trim to 240 chars
    return content[:240]

@app.post("/webhook")
async def handle_mention(data: dict):
    tweet = data.get("tweet_create_events", [{}])[0]
    if not tweet:
        return JSONResponse(status_code=200, content={"message": "No tweet data"})

    text     = tweet.get("text", "")
    user     = tweet.get("user", {}).get("screen_name", "")
    tweet_id = tweet.get("id_str", "")

    # extract first $TOKEN
    token_name = next(
        (w[1:].upper() for w in text.split() if w.startswith("$") and len(w) > 1),
        None
    )
    if not token_name:
        return JSONResponse(status_code=200, content={"message": "No token mentioned"})

    try:
        token_data = fetch_token_data(token_name)
        reply_text = generate_reply(token_data, token_name, user)

        # post as a threaded reply
        x_client.create_tweet(
            text=reply_text,
            in_reply_to_tweet_id=int(tweet_id)
        )
        return JSONResponse(status_code=200, content={"message": "Replied to mention"})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in handle_mention: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Degen Meme Bot is live. Mention me with a $TOKEN!"}
