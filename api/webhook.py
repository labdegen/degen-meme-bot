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
            return {"no_data": True}

        token_data = data.get("data", {}).get("token")
        price_data = data.get("data", {}).get("getTokenPrices", [{}])[0]

        if not token_data:
            logger.info(f"No token data for address: {address}")
            return {"no_data": True}

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
        return {"no_data": True}

def generate_reply(token_data: dict, query: str, tweet_text: str) -> str:
    """Call Grok for a ≤240-char reply, using token data or sentiment analysis."""
    system = (
        "You’re a degenerate crypto gambler, snarky but useful. Keep it ≤240 chars. "
        "For tokens, mock pumps/rugs, use price, supply, scam flag. "
        "For general text, analyze sentiment or reply conversationally."
    )
    if not token_data.get("no_data"):
        scam_note = " (SCAM ALERT!)" if token_data["isScam"] else ""
        user_msg = (
            f"Token: {query} ({token_data['symbol']}{scam_note}). "
            f"Price: ${token_data['priceUsd']} (Conf: {token_data['confidence']}). "
            f"Supply: {token_data['totalSupply']}, Circ: {token_data['circulatingSupply']}. "
            "Give a punchy degen take."
        )
    else:
        user_msg = (
            f"Tweet: {tweet_text}. No token data found. "
            "Analyze sentiment on X or reply conversationally as a crypto degen."
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
                if HEX_REGEX.match(tok):
                    addr = tok
                    query = tok
                else:
                    addr = SYMBOL_ADDRESS_MAP.get(tok.upper())
                    query = "$" + tok.upper()
                    if not addr:
                        token_data = {"no_data": True}
                        reply_content = generate_reply(token_data, query, txt)
                if addr:
                    token_data = fetch_token_data(addr)
                    reply_content = generate_reply(token_data, query, txt)
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
            if HEX_REGEX.match(tok):
                addr = tok
                query = tok
            else:
                addr = SYMBOL_ADDRESS_MAP.get(tok.upper())
                query = "$" + tok.upper()
                if not addr:
                    # Treat unmapped token as general text
                    token_data = {"no_data": True}
                    reply_content = generate_reply(token_data, query, txt)
            if addr:
                token_data = fetch_token_data(addr)
                reply_content = generate_reply(token_data, query, txt)
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