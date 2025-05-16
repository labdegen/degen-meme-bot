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

# Twitter X client setup
grok_url = "https://api.x.ai/v1/chat/completions"
perplexity_url = "https://api.perplexity.ai/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

api_key = os.getenv("X_API_KEY")
api_key_secret = os.getenv("X_API_KEY_SECRET")
access_token = os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("X_BEARER_TOKEN")
grok_key = os.getenv("GROK_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

# Redis
db = redis.Redis(
    host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"), decode_responses=True
)
db.ping(); logger.info("Redis connected")

# Initialize Tweepy
x_client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)
me = x_client.get_me().data
BOT_ID = me.id
logger.info(f"Auth as {me.username} (ID: {BOT_ID})")

# Constants
REDIS_PREFIX = "degen:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
ADDR_RE = re.compile(r'^[A-Za-z0-9]{43,44}$')

# Degen knowledge
DEGEN_KB = [
    "🚀 First $DEGEN on pump.fun (March 2024)",
    "🤝 Organic community token",
    "🎮 Play at jeetsvsdegens.com"
]

# Grok helper: never mention Base chain
def ask_grok(sys_prompt, usr_prompt, max_tokens=200):
    sys_prompt = sys_prompt + " ALWAYS refer to $DEGEN as the Solana token at the Solana contract address, never mention Base or Base chain."
    body = {"model":"grok-3","messages":[{"role":"system","content":sys_prompt},{"role":"user","content":usr_prompt}],"max_tokens":max_tokens,"temperature":0.7}
    h = {"Authorization":f"Bearer {grok_key}","Content-Type":"application/json"}
    r = requests.post(grok_url,json=body,headers=h,timeout=15); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# Perplexity helper: enforce same
def ask_perplexity(sys_prompt, usr_prompt, max_tokens=200):
    sys_prompt = sys_prompt + " ALWAYS refer to $DEGEN as the Solana token, never mention Base or Base chain."
    payload = {'model':'sonar-pro','messages':[{'role':'system','content':sys_prompt},{'role':'user','content':usr_prompt or ''}],
               'max_tokens':max_tokens,'temperature':1.0,'top_p':0.9,'search_recency_filter':'week'}
    h = {'Authorization':f'Bearer {perplexity_key}','Content-Type':'application/json'}
    r = requests.post(perplexity_url,json=payload,headers=h,timeout=60); r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()

# Fetch live DexScreener data
def fetch_data(addr):
    key = f"{REDIS_PREFIX}dex:{addr}"
    if c := db.get(key): return json.loads(c)
    r = requests.get(f"{DEXS_URL}{addr}",timeout=10); r.raise_for_status()
    d = r.json()[0]; t = d.get('baseToken',{})
    out = {'symbol':t.get('symbol'),'price_usd':float(d.get('priceUsd',0)),
           'volume_usd':float(d.get('volume',{}).get('h24',0)),
           'market_cap':float(d.get('marketCap',0)),
           'change_1h':float(d.get('priceChange',{}).get('h1',0)),
           'change_24h':float(d.get('priceChange',{}).get('h24',0)),
           'project_url':t.get('projectUrl',''),'socials':t.get('socials',[])}
    db.setex(key,300,json.dumps(out)); return out

def resolve_token(q):
    s = q.upper().lstrip('$')
    if s=='DEGEN': return 'DEGEN',DEGEN_ADDR
    if ADDR_RE.match(s): return None,s
    try:
        r = requests.get(f"https://api.dexscreener.com/latest/dex/search?search={s}",timeout=10); r.raise_for_status()
        for it in r.json():
            if it.get('chainId')=='solana':
                sym=it.get('baseToken',{}).get('symbol'); addr=it.get('pairAddress') or it.get('baseToken',{}).get('address')
                return sym,addr
    except: pass
    out=ask_grok('Map Solana symbol to address. Return JSON {"symbol":str,"address":str}.',f"Symbol: {s}",100)
    try: j=json.loads(out); return j.get('symbol'),j.get('address')
    except: return None,None

async def handle_mention(ev):
    txt=ev['tweet_create_events'][0]['text'].replace('@askdegen','').strip(); tid=ev['tweet_create_events'][0]['id_str']
    toks=[w for w in txt.split() if w.startswith('$') or ADDR_RE.match(w)]
    if toks:
        tok,q=toks[0],toks[0]
        sym,addr=resolve_token(tok)
        if addr:
            d=fetch_data(addr)
            if txt.strip()==tok:
                lines=[f"🚀 {d['symbol']} | ${d['price_usd']:,.6f}",f"MC ${d['market_cap']:,.0f}K | Vol24 ${d['volume_usd']:,.1f}K",f"1h {'🟢' if d['change_1h']>=0 else '🔴'}{d['change_1h']:+.2f}% | 24h {'🟢' if d['change_24h']>=0 else '🔴'}{d['change_24h']:+.2f}%"]
                if u:=d['project_url']: lines.append(f"🌐 {u}"); lines+= [f"{s['name']}: {s['url']}" for s in d['socials'] if s.get('name') and s.get('url')]
                reply='\n'.join(lines)
            else:
                sys=f"Expert Solana analyst: given these metrics {json.dumps(d)}, craft a concise (<240 chars) conversational reply."
                reply=ask_perplexity(sys,txt,150)
        else:
            reply=ask_perplexity("Crypto details unavailable—one concise tweet.",txt,80)
    else:
        reply=ask_grok("Answer as Tim Dillon, witty & direct, one tweet (<240 chars).",txt,120)
    x_client.create_tweet(text=reply[:240],in_reply_to_tweet_id=int(tid)); return {'message':'ok'}

async def degen_hourly_loop():
    while True:
        try:
            d=fetch_data(DEGEN_ADDR)
            sys=("Write exactly 4 sentences, positive & engaging about $DEGEN on Solana, "
                 f"using metrics price=${d['price_usd']:,.6f}, mc=${d['market_cap']:,.0f}K, vol24=${d['volume_usd']:,.1f}K. "
                 "Return only tweet text up to 280 chars.")
            try: promo=ask_perplexity(sys,"",200)[:280]
            except: promo=(f"$DEGEN trading=${d['price_usd']:,.6f}, mc=${d['market_cap']:,.0f}K, vol24=${d['volume_usd']:,.1f}K, 1h {d['change_1h']:+.2f}%,24h {d['change_24h']:+.2f}%")
            if promo:
                try: x_client.create_tweet(text=promo); logger.info("Promo v2 sent")
                except: tweepy.API(tweepy.OAuth1UserHandler(api_key,api_key_secret,access_token,access_token_secret)).update_status(promo); logger.info("Promo v1 sent")
        except Exception as e:
            logger.error(f"Promo loop error: {e}")
        await asyncio.sleep(3600)

async def poll_loop():
    while True:
        last=db.get(f"{REDIS_PREFIX}last_tweet_id"); since=int(last) if last else None
        res=x_client.get_users_mentions(id=BOT_ID, since_id=since, tweet_fields=['id','text','author_id'], expansions=['author_id'], user_fields=['username'], max_results=10)
        if res and res.data:
            us={u.id:u.username for u in res.includes.get('users',[])}
            for tw in reversed(res.data):
                ev={'tweet_create_events':[{'id_str':str(tw.id),'text':tw.text,'user':{'screen_name':us.get(tw.author_id,'?')}}]}
                try: await handle_mention(ev)
                except Exception as e: logger.error(f"Mention error: {e}")
                db.set(f"{REDIS_PREFIX}last_tweet_id",tw.id); db.set(f"{REDIS_PREFIX}last_mention",int(time.time()))
        lm=db.get(f"{REDIS_PREFIX}last_mention"); await asyncio.sleep(90 if lm and time.time()-int(lm)<3600 else 1800)

@app.on_event('startup')
async def startup():
    asyncio.create_task(poll_loop()); asyncio.create_task(degen_hourly_loop())

@app.get('/')
async def root(): return {'msg':'Bot live'}

@app.post('/test')
async def test_bot(r:Request): data=await r.json(); ev={'tweet_create_events':[{'id_str':'0','text':data.get('text',''),'user':{'screen_name':'test'}}]}; return await handle_mention(ev)
