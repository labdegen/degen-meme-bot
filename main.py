from fastapi import FastAPI, HTTPException, Request
import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import json
import asyncio
import time

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app
app = FastAPI()

# env
load_dotenv()
for v in (
    "X_API_KEY","X_API_KEY_SECRET",
    "X_ACCESS_TOKEN","X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY","PERPLEXITY_API_KEY",
    "REDIS_HOST","REDIS_PORT","REDIS_PASSWORD"
):
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var {v}")

# Twitter/X clients
api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("X_BEARER_TOKEN")

x_client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key, consumer_secret=api_key_secret,
    access_token=access_token, access_token_secret=access_token_secret
)
me = x_client.get_me().data
BOT_ID = me.id
logger.info(f"Authenticated as {me.username} (ID {BOT_ID})")

# legacy v1 for fallback
oauth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
x_api = tweepy.API(oauth)

# Redis
db = redis.Redis(
    host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"), decode_responses=True
)
db.ping()

# constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR   = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE      = re.compile(r"^[A-Za-z0-9]{43,44}$")

# endpoints & keys
GROK_URL        = "https://api.x.ai/v1/chat/completions"
PERP_URL        = "https://api.perplexity.ai/chat/completions"
DEXS_URL        = "https://api.dexscreener.com/token-pairs/v1/solana/"
grok_key        = os.getenv("GROK_API_KEY")
perplexity_key  = os.getenv("PERPLEXITY_API_KEY")

# helpers
def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int=200) -> str:
    body = {
        "model":"grok-3",
        "messages":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "max_tokens":max_tokens,
        "temperature":0.7
    }
    headers = {"Authorization":f"Bearer {grok_key}","Content-Type":"application/json"}
    r = requests.post(GROK_URL, json=body, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int=200) -> str:
    payload = {
        "model":"sonar-pro",
        "messages":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt or ""}
        ],
        "max_tokens":max_tokens,
        "temperature":1.0,
        "top_p":0.9,
        "search_recency_filter":"week"
    }
    headers = {"Authorization":f"Bearer {perplexity_key}","Content-Type":"application/json"}
    r = requests.post(PERP_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def fetch_data(addr: str) -> dict:
    key = f"{REDIS_PREFIX}dex:{addr}"
    if c := db.get(key):
        return json.loads(c)
    r = requests.get(f"{DEXS_URL}{addr}", timeout=10)
    r.raise_for_status()
    d = r.json()[0]
    t = d.get("baseToken",{})
    out = {
        "symbol":     t.get("symbol"),
        "price_usd":  float(d.get("priceUsd", 0)),
        "volume_usd": float(d.get("volume",{}).get("h24", 0)),
        "market_cap": float(d.get("marketCap",0)),
        "change_1h":  float(d.get("priceChange",{}).get("h1",0)),
        "change_24h": float(d.get("priceChange",{}).get("h24",0)),
        "link":       f"https://dexscreener.com/solana/{addr}"
    }
    db.setex(key,300,json.dumps(out))
    return out

def resolve_token(q: str) -> tuple[str,str]:
    s = q.upper().lstrip("$")
    if s=="DEGEN":
        return "DEGEN", DEGEN_ADDR
    if ADDR_RE.match(s):
        return None, s
    # dex search
    try:
        r = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}", timeout=10)
        r.raise_for_status()
        for it in r.json():
            if it.get("chainId")=="solana":
                sym = it["baseToken"]["symbol"]
                addr= it.get("pairAddress") or it["baseToken"]["address"]
                return sym, addr
    except:
        pass
    # fallback grok
    out = ask_grok(
        "Map Solana symbol to its contract address. Return JSON {'symbol':str,'address':str}.",
        f"Symbol: {s}",
        100
    )
    try:
        j = json.loads(out)
        return j.get("symbol"), j.get("address")
    except:
        return None, None

async def handle_mention(ev: dict):
    txt = ev["tweet_create_events"][0]["text"].replace("@askdegen","").strip()
    tid = ev["tweet_create_events"][0]["id_str"]
    tok = next((w for w in txt.split() if w.startswith("$") or ADDR_RE.match(w)), None)
    if tok:
        sym, addr = resolve_token(tok)
        if addr:
            d = fetch_data(addr)
            # pure preview
            if txt.strip()==tok:
                lines = [
                    f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",
                    f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",
                    f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"
                ]
                lines.append(d["link"])
                reply = "\n".join(lines)
            else:
                sys = f"Expert crypto professor: with metrics {json.dumps(d)}, draft a concise (<240 chars) conversational reply."
                reply = ask_perplexity(sys, txt, 150)
        else:
            reply = ask_perplexity("Crypto details unavailable—one concise tweet.", txt, 80)
    else:
        # freeform via Grok
        reply = ask_grok("Professional crypto professor: concise analytical response.", txt, 120)

    # reply
    x_client.create_tweet(text=reply[:240], in_reply_to_tweet_id=int(tid))
    return {"message":"ok"}

async def degen_hourly_loop():
    while True:
        d = fetch_data(DEGEN_ADDR)
        system = (
            "Professional crypto professor: write exactly 4 sentences that are positive, "
            f"engaging, and community-focused about $DEGEN on Solana, using price=${d['price_usd']:,.6f}, "
            f"mc=${d['market_cap']:,.0f}K, vol24=${d['volume_usd']:,.1f}K."
        )
        try:
            promo = ask_perplexity(system, "", 200)[:280]
        except Exception as e:
            logger.error(f"Perplexity failed ({e}), falling back to Grok")
            promo = ask_grok(system, "", 200)[:280]

        try:
            x_client.create_tweet(text=promo)
            logger.info("Hourly promo sent via v2")
        except Exception:
            x_api.update_status(promo)
            logger.info("Hourly promo sent via v1 fallback")

        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        last = db.get(f"{REDIS_PREFIX}last_tweet_id")
        since = int(last) if last else None
        res = x_client.get_users_mentions(
            id=BOT_ID, since_id=since,
            tweet_fields=["id","text","author_id"],
            expansions=["author_id"],
            user_fields=["username"],
            max_results=10
        )
        if res and res.data:
            users = {u.id:u.username for u in res.includes.get("users",[])}
            for tw in reversed(res.data):
                ev = {"tweet_create_events":[{
                    "id_str":str(tw.id),
                    "text":tw.text,
                    "user":{"screen_name":users.get(tw.author_id,"?")}
                }]}
                try:
                    await handle_mention(ev)
                except Exception as e:
                    logger.error(f"Mention error: {e}")
                db.set(f"{REDIS_PREFIX}last_tweet_id", tw.id)
                db.set(f"{REDIS_PREFIX}last_mention", int(time.time()))
        lm = db.get(f"{REDIS_PREFIX}last_mention")
        await asyncio.sleep(90 if lm and time.time()-int(lm)<3600 else 1800)

@app.on_event("startup")
async def startup():
    asyncio.create_task(poll_loop())
    asyncio.create_task(degen_hourly_loop())

@app.get("/")
async def root():
    return {"message":"Degen Meme Bot is live."}

@app.post("/test")
async def test_bot(r: Request):
    b = await r.json()
    ev = {"tweet_create_events":[{
        "id_str":"0","text":b.get("text",""),"user":{"screen_name":"test"}
    }]}
    return await handle_mention(ev)
