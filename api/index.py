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

# Initialize X API client
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Grok API endpoint
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# A simple symbol→address map for demo
SYMBOL_ADDRESS_MAP = {
    "DEGEN": "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
}

def fetch_token_data(symbol: str, network_id: int = 101):
    """
    TEST STUB: Instead of calling Codex.io, we return fake but plausible data.
    In production, swap this out for your real fetch_token_data() logic.
    """
    address = SYMBOL_ADDRESS_MAP.get(symbol, symbol)
    logger.info(f"Using address {address} for symbol {symbol}")

    # Return a dummy payload matching what your reply logic expects
    return {
        "id": address,
        "address": address,
        "name": symbol + " Token",
        "symbol": symbol,
        "totalSupply": "1,000,000,000",
        "info": {"circulatingSupply": "750,000,000"}
    }

def analyze_sentiment(token_name: str) -> str:
    """Get sentiment analysis from Grok with degen tone."""
    prompt = f"""
Yo, Grok, I'm a degen trader chasing meme coin pumps. Scan X for posts about {token_name} from the last 24h.
Give me a sentiment breakdown (bullish, bearish, neutral) in a table, with key comments and handles.
Keep it snarky, like you're mocking my YOLO bets but still delivering the goods.
"""
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.9
    }

    try:
        resp = requests.post(GROK_URL, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Grok API error: {e}")
        raise HTTPException(status_code=500, detail=f"Grok error: {e}")

def post_tweet(text: str):
    """Post a tweet with rate limit handling."""
    try:
        x_client.create_tweet(text=text[:280])
        logger.info("Tweet posted")
    except tweepy.TooManyRequests:
        logger.warning("X API rate limit hit")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as e:
        logger.error(f"X API error: {e}")
        raise HTTPException(status_code=500, detail=f"X API error: {e}")

@app.post("/webhook")
async def handle_mention(data: dict):
    """Handle incoming X mentions via webhook."""
    tweet = data.get("tweet_create_events", [{}])[0]
    if not tweet:
        return JSONResponse(status_code=200, content={"message": "No tweet data"})

    text = tweet.get("text", "").lower()
    user = tweet.get("user", {}).get("screen_name", "degen")
    # note: tweet_id unused for now, but could be used for replying in-thread
    tweet_id = tweet.get("id_str", "")

    # Extract token symbol (e.g., $PEPE, $DEGEN)
    token_name = None
    for word in text.split():
        if word.startswith("$") and len(word) > 1:
            token_name = word[1:].upper()
            break

    if not token_name:
        return JSONResponse(status_code=200, content={"message": "No token mentioned"})

    try:
        # Use our stubbed token data
        token_data = fetch_token_data(token_name)
        sentiment = analyze_sentiment(token_name)

        # Craft snarky reply
        reply = (
            f"@{user} ${token_name}? Oh, you’re late to the pump, huh?  \n"
            f"X vibe: {sentiment.splitlines()[0]}...  \n"
            f"{token_data['name']} is at {token_data['totalSupply']} total supply, "
            "but this chart’s a drunk rollercoaster.  \n"
            "Bet big, lose big, champ. DM for deeper dives or keep YOLOing into the abyss."
        )

        post_tweet(reply)
        return JSONResponse(status_code=200, content={"message": "Replied to mention"})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error handling mention: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Degen Meme Bot is live. Mention me with a $TOKEN!"}
