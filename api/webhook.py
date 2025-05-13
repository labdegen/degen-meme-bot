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

# Tweepy client (@askdegen creds)
x_client = tweepy.Client(
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

# Grok config
GROK_URL     = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Simple symbol→address map (stub)
SYMBOL_ADDRESS_MAP = {"DEGEN": "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"}
HEX_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

def fetch_token_data(address: str) -> dict:
    """Stub: replace with real Codex.io or CoinGecko call."""
    logger.info(f"Fetching data for {address}")
    return {
        "address": address,
        "name": f"Token@{address[:6]}…",
        "totalSupply": "1,000,000,000",
        "circulatingSupply": "750,000,000"
    }

def generate_reply(token_data: dict, query: str, user: str) -> str:
    """Call Grok for a unique, ≤240-char degen analysis."""
    system = (
        "You’re a degenerate crypto gambler. Keep it ≤240 chars, "
        "snarky but useful, mocking pumps and rugs, include one data point."
    )
    user_msg = (
        f"User @{user} asked about {query}. "
        f"Supply={token_data['totalSupply']}, circ={token_data['circulatingSupply']}. "
        "Give a punchy degen take."
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

    r = requests.post(GROK_URL, json=body, headers=headers)
    if r.status_code != 200:
        logger.error("Grok failed: %s", r.text)
        raise HTTPException(status_code=500, detail="Grok analysis failed")
    text = r.json()["choices"][0]["message"]["content"].strip()
    return text[:240]

@app.post("/")
async def handle_mention(data: dict):
    evt = data.get("tweet_create_events", [{}])[0]
    if not evt:
        return JSONResponse({"message":"No tweet data"}, status_code=200)

    txt      = evt.get("text","")
    user     = evt.get("user",{}).get("screen_name","")
    tid      = evt.get("id_str","")
    # find $TOKEN
    tok = next((w[1:] for w in txt.split() if w.startswith("$") and len(w)>1), None)
    if not tok:
        return JSONResponse({"message":"No token mentioned"}, status_code=200)

    # decide if address or symbol
    if HEX_REGEX.match(tok):
        addr = tok
        query = tok
    else:
        addr  = SYMBOL_ADDRESS_MAP.get(tok.upper(), tok)
        query = "$" + tok.upper()

    try:
        token_data = fetch_token_data(addr)
        reply = generate_reply(token_data, query, user)

        # try in-thread, fallback standalone
        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
        except tweepy.errors.Forbidden:
            logger.warning("Threaded reply failed; posting standalone")
            x_client.create_tweet(text=reply)

        return JSONResponse({"message":"Replied to mention"}, status_code=200)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("handle_mention error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
