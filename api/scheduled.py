import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

def create_scheduled_post():
    system = (
        "Crypto analyst. Summarize meme coin market trends, vision for future. ≤750 chars, use only what's needed. "
        "Fun, witty, mention trending coins. JSON: {'post': str}"
    )
    user_msg = "Analyze X for meme coin trends. Predict future, highlight top coins."
    headers = {"Authorization": f"Bearer {os.getenv('GROK_API_KEY')}", "Content-Type": "application/json"}
    body = {"model": "grok-3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 750, "temperature": 0.7}
    try:
        r = requests.post("https://api.x.ai/v1/chat/completions", json=body, headers=headers, timeout=15)
        data = json.loads(r.json()["choices"][0]["message"]["content"].strip())
        post = data.get("post", "Meme coins soar! $WIF’s 90% bullish on X. Future: More games, utility!")[:750]
        x_client.create_tweet(text=post)
        logger.info(f"Posted: {post}")
    except Exception as e:
        logger.error(f"Scheduled post error: {e}")

if __name__ == "__main__":
    create_scheduled_post()