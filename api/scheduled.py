from fastapi import FastAPI, HTTPException
import requests
import tweepy
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

def find_trending_tokens():
    """Find trending meme coins via Grok."""
    prompt = """
    Scan X for meme coins trending in the last 4h. List the top 3 with sentiment and volume spikes,
    in a degen trader tone. Keep it snarky and short.
    """
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.9
    }
    
    try:
        response = requests.post(GROK_URL, json=body, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Grok API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find trends: {str(e)}")

def post_tweet(text):
    """Post a tweet with rate limit handling."""
    try:
        x_client.create_tweet(text=text[:280])
    except tweepy.TooManyRequests:
        logger.warning("X API rate limit hit")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as e:
        logger.error(f"X API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to post tweet: {str(e)}")

@app.get("/scheduled")
async def scheduled_post():
    """Post trending tokens every 4 hours."""
    try:
        trends = find_trending_tokens()
        tweet = f"""
        🚨 Degen Alert 🚨 {trends[:200]}...
        Moon or rug? Place your bets, degens! #MemeCoins
        """
        post_tweet(tweet)
        return {"message": "Scheduled post sent"}
    except Exception as e:
        logger.error(f"Scheduled post error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})