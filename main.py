from fastapi import FastAPI
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
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load env vars and validate
load_dotenv()
required_vars = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "X_BEARER_TOKEN",
    "GROK_API_KEY", "PERPLEXITY_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]
for v in required_vars:
    if not os.getenv(v):
        raise RuntimeError(f"Missing env var: {v}")

# API endpoints
GROK_URL       = "https://api.x.ai/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEXS_URL       = "https://api.dexscreener.com/token-pairs/v1/solana/"
SEARCH_URL     = "https://api.dexscreener.com/latest/dex/search?search={}"

# Credentials
API_KEY             = os.getenv("X_API_KEY")
API_KEY_SECRET      = os.getenv("X_API_KEY_SECRET")
ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
BEARER_TOKEN        = os.getenv("X_BEARER_TOKEN")
GROK_KEY            = os.getenv("GROK_API_KEY")
PERPLEXITY_KEY      = os.getenv("PERPLEXITY_API_KEY")

# Redis client
db = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
db.ping()
logger.info("Redis connected")

# Initialize Tweepy clients
x_client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_KEY_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)
me = x_client.get_me().data
BOT_ID = me.id
logger.info(f"Authenticated as: {me.username} (ID: {BOT_ID})")

oauth = tweepy.OAuth1UserHandler(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
x_api = tweepy.API(oauth)

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR   = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE      = re.compile(r'^[A-Za-z0-9]{43,44}$')

# === RATE LIMIT GUARDS ===
RATE_WINDOW     = 15 * 60  # 15 minutes in seconds
MENTIONS_LIMIT  = 10       # mentions lookups per window
TWEETS_LIMIT    = 50       # tweet posts per window

mentions_timestamps = deque()
tweet_timestamps    = deque()

async def safe_mention_lookup(fn, *args, **kwargs):
    # Prevent >MENTIONS_LIMIT calls per RATE_WINDOW
    now = time.time()
    while mentions_timestamps and now - mentions_timestamps[0] > RATE_WINDOW:
        mentions_timestamps.popleft()
    if len(mentions_timestamps) >= MENTIONS_LIMIT:
        wait = RATE_WINDOW - (now - mentions_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Mentions limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        res = fn(*args, **kwargs)
    except tweepy.TooManyRequests as e:
        # honor retry-after
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] get_users_mentions 429; back off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_mention_lookup(fn, *args, **kwargs)
    mentions_timestamps.append(time.time())
    return res

async def safe_tweet(text: str, **kwargs):
    # Prevent >TWEETS_LIMIT tweets per RATE_WINDOW
    now = time.time()
    while tweet_timestamps and now - tweet_timestamps[0] > RATE_WINDOW:
        tweet_timestamps.popleft()
    if len(tweet_timestamps) >= TWEETS_LIMIT:
        wait = RATE_WINDOW - (now - tweet_timestamps[0]) + 1
        logger.warning(f"[RateGuard] Tweet limit reached; sleeping {wait:.0f}s")
        await asyncio.sleep(wait)
    try:
        resp = x_client.create_tweet(text=text, **kwargs)
    except tweepy.TooManyRequests as e:
        reset = int(e.response.headers.get('x-rate-limit-reset', time.time() + RATE_WINDOW))
        wait = max(0, reset - time.time()) + 1
        logger.error(f"[RateGuard] create_tweet 429; back off {wait:.0f}s")
        await asyncio.sleep(wait)
        return await safe_tweet(text=text, **kwargs)
    tweet_timestamps.append(time.time())
    return resp

# === HELPER FUNCTIONS ===

def ask_grok(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    body = {"model":"grok-3","messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],"max_tokens":max_tokens,"temperature":0.7}
    headers = {"Authorization":f"Bearer {GROK_KEY}","Content-Type":"application/json"}
    try:
        r = requests.post(GROK_URL, json=body, headers=headers, timeout=35)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Grok fallback: {e}")
        return "Stack more Degen."


def ask_perplexity(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    payload={
        'model':'sonar-pro','messages':[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],
        'max_tokens':max_tokens,'temperature':1.0,'top_p':0.9,'search_recency_filter':'week'
    }
    headers={'Authorization':f'Bearer {PERPLEXITY_KEY}','Content-Type':'application/json'}
    try:
        resp = requests.post(PERPLEXITY_URL,json=payload,headers=headers,timeout=30)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Perplexity fallback: {e}")
        return ask_grok("You are a professional market analyst.",user_prompt,max_tokens)


def fetch_data(addr:str)->dict:
    key=f"{REDIS_PREFIX}dex:{addr}"
    if cached:=db.get(key): return json.loads(cached)
    r=requests.get(f"{DEXS_URL}{addr}",timeout=10);r.raise_for_status()
    data=r.json()[0];base=data.get('baseToken',{})
    out={
        'symbol':base.get('symbol'),'price_usd':float(data.get('priceUsd',0)),
        'volume_usd':float(data.get('volume',{}).get('h24',0)),
        'market_cap':float(data.get('marketCap',0)),'change_1h':float(data.get('priceChange',{}).get('h1',0)),
        'change_24h':float(data.get('priceChange',{}).get('h24',0)),'link':f"https://dexscreener.com/solana/{addr}",
        'logo':base.get('logoURI'),'address':addr
    }
    db.setex(key,300,json.dumps(out));return out


def resolve_token(q:str)->tuple:
    s=q.upper().lstrip('$')
    if s=='DEGEN': return 'DEGEN',DEGEN_ADDR
    if ADDR_RE.match(s): return None,s
    try:
        resp=requests.get(SEARCH_URL.format(s),timeout=10);resp.raise_for_status()
        for item in resp.json():
            if item.get('chainId')=='solana':
                base=item.get('baseToken',{});addr=item.get('pairAddress') or base.get('address')
                if addr: return base.get('symbol'),addr
    except: logger.warning("Search fallback failed")
    try:
        prompt=f"Find the Solana token contract address for the symbol {s}. Return JSON {{'symbol': str,'address': str}}."
        out=ask_perplexity("You are a Solana token resolver.",prompt)
        data=json.loads(out);return data.get('symbol'),data.get('address')
    except: return None,None


def format_metrics(data:dict)->str:
    return(f"🚀 {data['symbol']} | ${data['price_usd']:,.6f}\nMC ${data['market_cap']:,.0f} | Vol24 ${data['volume_usd']:,.0f}\n"
            f"1h {'🟢' if data['change_1h']>=0 else '🔴'}{data['change_1h']:+.2f}% | 24h {'🟢' if data['change_24h']>=0 else '🔴'}{data['change_24h']:+.2f}%\n{data['link']}")


def format_convo_reply(data:dict,question:str)->str:
    prompt=(f"Here are the token metrics: Symbol: {data['symbol']}, Price: ${data['price_usd']:,.6f}, Market Cap: ${data['market_cap']:,.0f}, "
            f"24h Change: {data['change_24h']:+.2f}%, Volume: ${data['volume_usd']:,.0f}. Answer this in 230 chars ending with 'NFA': {question}")
    return ask_grok("You are a crypto analyst replying to questions.",prompt,120)

# === FASTAPI ENDPOINTS & LOOPS ===
@app.get("/")
async def root(): return{"status":"Degen bot is live."}

@app.on_event("startup")
async def startup_event():
    await asyncio.sleep(60)
    asyncio.create_task(poll_loop())
    await asyncio.sleep(3600)
    asyncio.create_task(hourly_post_loop())

async def poll_loop():
    while True:
        last=db.get(f"{REDIS_PREFIX}last_tweet_id")
        since_id=int(last) if last else None
        res=await safe_mention_lookup(
            x_client.get_users_mentions,id=BOT_ID,since_id=since_id,
            tweet_fields=['id','text','author_id'],expansions=['author_id'],
            user_fields=['username'],max_results=10
        )
        if res and res.data:
            users={u.id:u.username for u in res.includes.get('users',[])}
            for tw in reversed(res.data):
                ev={'tweet_create_events':[{'id_str':str(tw.id),'text':tw.text,'user':{'screen_name':users.get(tw.author_id,'?')}}]}
                try: await handle_mention(ev)
                except Exception as e: logger.error(f"Mention error: {e}")
                db.set(f"{REDIS_PREFIX}last_tweet_id",tw.id)
                db.set(f"{REDIS_PREFIX}last_mention",int(time.time()))
        await asyncio.sleep(110)  # poll every 110s

async def hourly_post_loop():
    while True:
        try:
            d=fetch_data(DEGEN_ADDR)
            card=format_metrics(d)
            context=ask_grok("You're a Degen community member summarizing recent metrics.",json.dumps(d),160)
            tweet=f"{card}\n{context}"
            if len(tweet)>380: tweet=tweet[:380].rsplit('.',1)[0]+'.'
            await safe_tweet(text=tweet)
            logger.info("Hourly promo posted")
        except Exception as e:
            logger.error(f"Promo error: {e}")
        await asyncio.sleep(3600)

async def handle_mention(ev:dict):
    events=ev.get('tweet_create_events') or []
    if not events or not events[0].get('text'): return{'message':'no valid mention'}
    txt=events[0]['text'].replace('@askdegen','').strip()
    tid=events[0]['id_str']
    txt_up=txt.upper()
    # Keyword triggers
    if txt_up=='DEX':
        d=fetch_data(DEGEN_ADDR);metrics=format_metrics(d);logo=d.get('logo','')
        reply=f"{metrics}\n{logo}"
    elif txt_up=='CA':
        reply=f"Contract Address: {DEGEN_ADDR}"
    else:
        token=next((w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)),None)
        if token:
            sym,addr=resolve_token(token)
            if addr: reply=format_metrics(fetch_data(addr)) if token.strip()==txt else format_convo_reply(fetch_data(addr),txt)
            else: reply=ask_perplexity("You are a crypto researcher.",txt,160)
        else:
            reply=ask_grok("Professional crypto professor: concise analytical response.",txt,160)
    tweet=reply.strip()
    if len(tweet)>240:
        tweet=(tweet[:240].rsplit('.',1)[0]+'.') if '.' in tweet else (tweet[:240].rsplit(' ',1)[0]+'...')
    await safe_tweet(text=tweet,in_reply_to_tweet_id=int(tid))
    return{'message':'ok'}

# Allow direct run
if __name__=="__main__":
    import uvicorn
    port=int(os.getenv('PORT',10000))
    uvicorn.run('main:app',host='0.0.0.0',port=port)
