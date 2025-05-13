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
required_env_vars = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "GROK_API_KEY", "CODEX_API_KEY"]
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

# Codex.io config
CODEX_API_URL = "https://graph.codex.io/graphql"
CODEX_API_KEY = os.getenv("CODEX_API_KEY")

# Simple symbol→address map
SYMBOL_ADDRESS_MAP = {"DEGEN": "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"}
HEX_REGEX = re.compile(r"^(0x[a-fA-F0-9]{40}|[A-Za-z0-9]{43,44})$")  # Ethereum or Solana addresses

def fetch_token_data(address: str) -> dict:
    """Fetch token metadata and price from Codex.io GraphQL API."""
    logger.info(f"Fetching data for address: {address}")
    network_id = 101 if len(address) > 40 else 1  # Solana or Ethereum
    query = """
    query GetTokenData($input: TokenInput!, $priceInput: [GetPriceInput!]!) {
        token(input: $input) {
            address
            name
            symbol
            totalSupply
            decimals
            isScam
            info {
                circulatingSupply
            }
        }
        getTokenPrices(inputs: $priceInput) {
            address
            priceUsd
            confidence
        }
    }
    """
    body = {
        "query": query,
        "variables": {
            "input": {
                "address": address,
                "networkId": network_id
            },
            "priceInput": [
                {
                    "address": address,
                    "networkId": network_id
                }
            ]
        }
    }
    headers = {
        "Authorization": f"Bearer {CODEX_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(CODEX_API_URL, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Codex.io response: {data}")
        if "errors" in data:
            logger.error(f"Codex.io API error: {data['errors']}")
            return {"no_data": True, "message": "That token doesn’t deserve a reply"}

        token_data = data.get("data", {}).get("token")
        price_data = data.get("data", {}).get("getTokenPrices", [{}])[0]

        if not token_data:
            logger.info(f"No token data for address: {address}")
            return {"no_data": True, "message": "That token doesn’t deserve a reply"}

        return {
            "address": token_data["address"],
            "name": token_data.get("name", "Unknown"),
            "symbol": token_data.get("symbol", "Unknown"),
            "totalSupply": str(token_data.get("totalSupply", "0")),
            "circulatingSupply": str(token_data["info"].get("circulatingSupply", "0") if token_data.get("info") else "0"),
            "decimals": token_data.get("decimals", 0),
            "isScam": token_data.get("isScam", False),
            "priceUsd": str(price_data.get("priceUsd", "0")),
            "confidence": str(price_data.get("confidence", "0"))
        }
    except requests.RequestException as e:
        logger.error(f"Codex.io request error for address {address}: {str(e)}")
        return {"no_data": True, "message": "That token doesn’t deserve a reply"}

def generate_reply(token_data: dict, query: str, user: str) -> str:
    """Call Grok for a ≤240-char degen analysis with key data points."""
    if token_data.get("no_data"):
        return token_data["message"]
    system = (
        "You’re a degenerate crypto gambler. Keep it ≤240 chars, snarky but useful, "
        "mock pumps/rugs, use price, supply, and scam flag for insights."
    )
    scam_note = " (SCAM ALERT!)" if token_data["isScam"] else ""
    user_msg = (
        f"Token: {query} ({token_data['symbol']}{scam_note}). "
        f"Price: ${token_data['priceUsd']} (Conf: {token_data['confidence']}). "
        f"Supply: {token_data['totalSupply']}, Circ: {token_data['circulatingSupply']}. "
        "Give a punchy degen take."
    )
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 200,
        "temperature": 0.9
    }
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=10)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text[:240]
    except requests.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grok analysis failed: {str(e)}")

@app.post("/")
async def handle_mention(data: dict):
    """Handle Twitter webhook for @askdegen mentions with $TOKEN or contract address."""
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

    if not tok:
        logger.info(f"No token or address in tweet: {txt}")
        return JSONResponse({"message": "No token or address mentioned"}, status_code=200)

    # Determine address and query
    if HEX_REGEX.match(tok):
        addr = tok
        query = tok
    else:
        addr = SYMBOL_ADDRESS_MAP.get(tok.upper())
        query = "$" + tok.upper()
        if not addr:
            logger.error(f"No address mapped for token: {tok}")
            return JSONResponse({"message": f"No address found for ${tok}"}, status_code=400)

    try:
        token_data = fetch_token_data(addr)
        reply_content = generate_reply(token_data, query, user)
        reply = f"@{user} {reply_content}"
        max_reply_length = 280 - len(f"@{user} ") - 1
        reply = reply[:max_reply_length]

        try:
            x_client.create_tweet(text=reply, in_reply_to_tweet_id=int(tid))
            logger.info(f"Replied to @{user} with: {reply}")
        except tweepy.errors.Forbidden as e:
            logger.warning(f"Threaded reply failed for @{user}: {str(e)}; posting standalone")
            x_client.create_tweet(text=reply)
        except tweepy.TweepyException as e:
            logger.error(f"Failed to post reply to @{user}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

        return JSONResponse({"message": "Replied to mention"}, status_code=200)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"handle_mention error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)