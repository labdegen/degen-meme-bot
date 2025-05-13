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

# Initialize X API client (@askdegen creds)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Grok endpoint & key
GROK_URL     = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Stub map for symbols
SYMBOL_ADDRESS_MAP = {
    "DEGEN": "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
}

HEX_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

def fetch_token_data(address: str):
    """
    TEST STUB: Replace with your real Codex.io or CoinGecko fetch.
    """
    # If address came from a symbol, we already mapped it.
    logger.info(f"Fetching data for address {address}")
    return {
        "address": address,
        "name": f"Token@{address[:6]}…",
        "symbol": None,
        "totalSupply": "1,000,000,000",
        "circulatingSupply": "750,000,000"
    }

def generate_reply(token_data: dict, query: str, user_handle: str) -> str:
    """
    Ask Grok for a dynamic, ≤240-char degen analysis.
    """
    system = (
        "You are a degenerate crypto gambler. Keep it ≤240 chars, "
        "snarky but useful. Call out pumps and rugs, include one data point."
    )
    user_msg = (
        f"User @{user_handle} asked about {query}. "
        f"Data: totalSupply={token_data['totalSupply']}, "
        f"circulatingSupply={token_data['circulatingSupply']}. "
        "Give me a punchy degen take."
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
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
        logger.error("Grok error: %s", resp.text)
        raise HTTPException(status_code=500, detail="Grok analysis failed")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    return content[:240]

@app.post("/webhook")
async def handle_mention(data: dict):
    event = data.get("tweet_create_events", [{}])[0]
    if not event:
        return JSONResponse(status_code=200, content={"message":"No tweet data"})

    text     = event.get("text", "")
    user     = event.get("user", {}).get("screen_name", "")
    tweet_id = event.get("id_str", "")

    # Look for $WORD
    token_mention = next(
        (w[1:] for w in text.split() if w.startswith("$") and len(w) > 1),
        None
    )
    if not token_mention:
        return JSONResponse(status_code=200, content={"message":"No token mentioned"})

    # Determine if it's an address or a symbol
    query = token_mention
    if HEX_REGEX.match(token_mention):
        address = token_mention
    else:
        address = SYMBOL_ADDRESS_MAP.get(token_mention.upper())
        if not address:
            # If symbol not in map, treat the SYMBOL itself as query
            address = token_mention

    try:
        token_data = fetch_token_data(address)
        # Use the original mention ($DEGEN or $0x...) as the query label
        reply_text = generate_reply(token_data, "$" + token_mention, user)

        # Attempt threaded reply; fallback to standalone
        try:
            x_client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=int(tweet_id)
            )
        except tweepy.errors.Forbidden:
            logger.warning("Threaded reply failed; posting standalone.")
            x_client.create_tweet(text=reply_text)

        return JSONResponse(status_code=200, content={"message":"Replied to mention"})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("handle_mention error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message":"Degen Meme Bot is live. Mention me with a $TOKEN!"}
