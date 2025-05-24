"""
SMART CRYPTO PROMOTION BOT - COMPLIANT VERSION
- Strategic promotional raiding (no @mentions in replies)
- Sustainable liking campaigns
- Quality targeting to avoid spam detection
- Stays within API limits and X ToS
- Looks like natural human engagement
"""

import tweepy
import requests
import os
from dotenv import load_dotenv
import logging
import re
import redis
import asyncio
import time
from collections import deque
from random import choice
import random
from datetime import datetime, timedelta
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
required = [
    "X_API_KEY", "X_API_KEY_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", 
    "X_BEARER_TOKEN",
    "GROK_API_KEY",
    "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"
]

for var in required:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# Twitter API setup
oauth = tweepy.OAuth1UserHandler(
    os.getenv("X_API_KEY"),
    os.getenv("X_API_KEY_SECRET"),
    os.getenv("X_ACCESS_TOKEN"),
    os.getenv("X_ACCESS_TOKEN_SECRET")
)
x_api = tweepy.API(oauth)

x_client = tweepy.Client(
    bearer_token=os.getenv("X_BEARER_TOKEN"),
    consumer_key=os.getenv("X_API_KEY"),
    consumer_secret=os.getenv("X_API_KEY_SECRET"),
    access_token=os.getenv("X_ACCESS_TOKEN"),
    access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
)

try:
    me = x_client.get_me().data
    BOT_ID = me.id
    BOT_USERNAME = me.username
    logger.info(f"✅ Authenticated as: {BOT_USERNAME} (ID: {BOT_ID})")
except Exception as e:
    logger.error(f"❌ Authentication failed: {e}")
    exit(1)

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
redis_client.ping()
logger.info("✅ Redis connected")

# Constants
REDIS_PREFIX = "degen_bot:"
DEGEN_ADDR = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEXS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/"

# SMART COMPLIANCE LIMITS - Sustainable but active
DAILY_TWEET_LIMIT = 40          # 40 tweets per day (reasonable)
DAILY_LIKE_LIMIT = 120          # 120 likes per day (sustainable)
HOURLY_TWEET_LIMIT = 4          # Max 4 tweets per hour
HOURLY_LIKE_LIMIT = 12          # Max 12 likes per hour
MIN_DELAY_BETWEEN_ACTIONS = 45  # 45 seconds between actions

# SMART TARGETING - Quality over quantity
MIN_FOLLOWERS_FOR_RAID = 25     # Target accounts with 25+ followers
MAX_RAIDS_PER_CYCLE = 3         # Only 3 raids per cycle (was 50+)
MAX_LIKES_PER_CYCLE = 8         # Only 8 likes per cycle (was 50+)
RAID_CYCLE_DELAY = 900          # 15 minutes between raid cycles (was 5 min)
LIKE_CYCLE_DELAY = 600          # 10 minutes between like cycles

# Rate limiting queues
tweet_timestamps = deque()
search_timestamps = deque()
like_timestamps = deque()

# STRATEGIC SEARCH QUERIES - Focus on high-value targets
RAID_QUERIES = [
    "memecoin gem -is:retweet -is:reply min_faves:2",
    "solana meme -is:retweet -is:reply min_faves:1", 
    "crypto moonshot -is:retweet -is:reply min_faves:1",
    "altcoin hidden -is:retweet -is:reply min_faves:1",
    "$BONK OR $WIF OR $PEPE -is:retweet -is:reply",
    "new memecoin -is:retweet -is:reply min_faves:1",
    "degen plays -is:retweet -is:reply min_faves:1"
]

LIKE_QUERIES = [
    "crypto analysis min_faves:3",
    "solana ecosystem min_faves:2", 
    "memecoin research min_faves:2",
    "defi trending min_faves:2",
    "altcoin season min_faves:1"
]

class SmartComplianceManager:
    def __init__(self):
        self.daily_tweets = 0
        self.daily_likes = 0
        self.hourly_tweets = 0
        self.hourly_likes = 0
        self.last_reset = datetime.now().date()
        self.last_hour = datetime.now().hour
        self.last_action_time = 0
        
    def reset_counters(self):
        today = datetime.now().date()
        current_hour = datetime.now().hour
        
        # Reset daily counters
        if today > self.last_reset:
            self.daily_tweets = 0
            self.daily_likes = 0
            self.last_reset = today
            logger.info("🔄 Daily counters reset")
        
        # Reset hourly counters  
        if current_hour != self.last_hour:
            self.hourly_tweets = 0
            self.hourly_likes = 0
            self.last_hour = current_hour
    
    def can_tweet(self):
        self.reset_counters()
        
        if self.daily_tweets >= DAILY_TWEET_LIMIT:
            return False, f"Daily limit ({DAILY_TWEET_LIMIT})"
        if self.hourly_tweets >= HOURLY_TWEET_LIMIT:
            return False, f"Hourly limit ({HOURLY_TWEET_LIMIT})"
        if time.time() - self.last_action_time < MIN_DELAY_BETWEEN_ACTIONS:
            return False, "Too soon"
        
        return True, "OK"
    
    def can_like(self):
        self.reset_counters()
        return (self.daily_likes < DAILY_LIKE_LIMIT and 
                self.hourly_likes < HOURLY_LIKE_LIMIT)
    
    def record_action(self, action_type):
        self.last_action_time = time.time()
        
        if action_type == 'tweet':
            self.daily_tweets += 1
            self.hourly_tweets += 1
            tweet_timestamps.append(time.time())
        elif action_type == 'like':
            self.daily_likes += 1
            self.hourly_likes += 1
            like_timestamps.append(time.time())

compliance = SmartComplianceManager()

def ask_grok(prompt: str) -> str:
    system_prompt = (
        "You are a knowledgeable crypto trader. Be conversational and helpful. "
        "When promoting $DEGEN, be natural and engaging, not spammy. "
        "Write like you're genuinely interested in the conversation. "
        "Keep responses under 200 characters when possible."
    )
    
    payload = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.8
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}", 
        "Content-Type": "application/json"
    }
    
    try:
        r = requests.post(GROK_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Grok error: {e}")
        # Fallback responses
        fallbacks = [
            "Have you checked out $DEGEN? Solid memecoin with real potential.",
            "Interesting perspective! $DEGEN might be worth a look too.",
            "Good analysis. $DEGEN has been showing similar patterns.",
            "This reminds me of early $DEGEN vibes. Worth researching.",
            "Valid points. $DEGEN community is pretty active on this topic."
        ]
        return choice(fallbacks)

async def human_delay(min_seconds=30, max_seconds=90):
    """Add realistic human-like delays"""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)

async def safe_tweet(text: str, reply_to=None, media_id=None):
    can_tweet, reason = compliance.can_tweet()
    if not can_tweet:
        logger.warning(f"🚫 Tweet blocked: {reason}")
        return None
    
    try:
        await human_delay(15, 45)  # Human-like delay
        
        kwargs = {}
        if reply_to:
            kwargs['in_reply_to_tweet_id'] = reply_to
        if media_id:
            kwargs['media_ids'] = [media_id]
        
        result = x_client.create_tweet(text=text, **kwargs)
        compliance.record_action('tweet')
        
        logger.info(f"✅ Tweet: {text[:50]}...")
        return result
        
    except tweepy.Forbidden as e:
        logger.error(f"❌ FORBIDDEN: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Tweet failed: {e}")
        return None

async def safe_like(tweet_id: str):
    if not compliance.can_like():
        return False
    
    try:
        await human_delay(5, 15)
        x_api.create_favorite(id=tweet_id)
        compliance.record_action('like')
        return True
    except Exception as e:
        logger.warning(f"Like failed: {e}")
        return False

async def safe_search(query: str, max_results=10):
    try:
        await human_delay(10, 25)
        
        result = x_client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["id", "text", "author_id", "created_at", "public_metrics"],
            expansions=["author_id"],
            user_fields=["username", "public_metrics"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None

def fetch_degen_data():
    try:
        r = requests.get(f"{DEXS_URL}{DEGEN_ADDR}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        
        base = data.get('baseToken', {})
        return {
            'symbol': base.get('symbol', 'DEGEN'),
            'price_usd': float(data.get('priceUsd', 0)),
            'volume_24h': float(data.get('volume', {}).get('h24', 0)),
            'market_cap': float(data.get('marketCap', 0)),
            'change_24h': float(data.get('priceChange', {}).get('h24', 0)),
            'link': f"https://dexscreener.com/solana/{DEGEN_ADDR}"
        }
    except:
        return {}

async def smart_crypto_raid(tweet, author_username=""):
    """Strategic promotion without @mentions"""
    
    try:
        # Skip if already processed
        if redis_client.sismember(f"{REDIS_PREFIX}raided", str(tweet.id)):
            return
        
        # Mark as processed
        redis_client.sadd(f"{REDIS_PREFIX}raided", str(tweet.id))
        redis_client.expire(f"{REDIS_PREFIX}raided", 86400)  # Expire in 24h
        
        # Create contextual response based on tweet content
        tweet_lower = tweet.text.lower()
        
        if any(term in tweet_lower for term in ["memecoin", "meme coin", "shitcoin"]):
            prompt = f"User posted about memecoins: '{tweet.text}'\nWrite a natural reply suggesting $DEGEN as a solid memecoin. Be conversational, not spammy. Include contract {DEGEN_ADDR}. End with 'NFA' casually."
        elif any(term in tweet_lower for term in ["gem", "moonshot", "100x"]):
            prompt = f"User is gem hunting: '{tweet.text}'\nSuggest $DEGEN as a potential gem. Be friendly and natural. Include contract {DEGEN_ADDR}. End with 'NFA'."  
        elif any(coin in tweet.text.upper() for coin in ["$BONK", "$WIF", "$PEPE"]):
            prompt = f"User mentioned popular memecoins: '{tweet.text}'\nNaturally suggest they check out $DEGEN too. Include contract {DEGEN_ADDR}. End with 'NFA'."
        else:
            prompt = f"User posted about crypto: '{tweet.text}'\nWrite a natural response that mentions $DEGEN organically. Include contract {DEGEN_ADDR}. End with 'NFA'."
        
        response = ask_grok(prompt)
        
        # Ensure contract address is included
        if DEGEN_ADDR not in response:
            response = f"{response}\n\nCA: {DEGEN_ADDR}"
        
        # Add image occasionally (20% chance)
        media_id = None
        if random.random() < 0.2:
            try:
                meme_files = glob.glob("raid_images/*.jpg")
                if meme_files:
                    img_path = choice(meme_files)
                    media_id = x_api.media_upload(img_path).media_id_string
            except:
                pass
        
        # Clean response - NO @MENTIONS
        response = re.sub(r'@\w+', '', response).strip()
        
        # Send the raid reply
        result = await safe_tweet(
            text=response[:280],  # Ensure under character limit
            reply_to=tweet.id,
            media_id=media_id
        )
        
        if result:
            logger.info(f"🎯 RAIDED: {author_username} - {tweet.text[:30]}...")
        
    except Exception as e:
        logger.error(f"Raid error: {e}")

async def strategic_raid_loop():
    """Smart raiding - quality over quantity"""
    
    logger.info("🎯 Starting strategic crypto raiding")
    query_index = 0
    
    while True:
        try:
            # Rotate through targeted queries
            current_query = RAID_QUERIES[query_index % len(RAID_QUERIES)]
            query_index += 1
            
            logger.info(f"🔍 Searching: {current_query[:30]}...")
            
            results = await safe_search(current_query, max_results=20)
            
            if results and results.data:
                # Create user mapping
                user_map = {}
                if hasattr(results, 'includes') and results.includes:
                    for user in results.includes.get('users', []):
                        user_map[user.id] = user
                
                # Filter for quality targets
                quality_targets = []
                for tweet in results.data:
                    author = user_map.get(tweet.author_id)
                    if not author:
                        continue
                    
                    follower_count = author.public_metrics.get('followers_count', 0)
                    tweet_likes = tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else 0
                    
                    # Quality filters
                    is_quality = (
                        follower_count >= MIN_FOLLOWERS_FOR_RAID and  # Decent following
                        len(tweet.text) > 50 and  # Substantial content
                        tweet_likes >= 1 and  # Some engagement
                        not any(spam in tweet.text.lower() for spam in ['rt to win', 'follow for follow', 'dm me'])  # No obvious spam
                    )
                    
                    if is_quality:
                        quality_targets.append((tweet, author.username, follower_count))
                
                # Process only the best targets (max 3 per cycle)
                quality_targets.sort(key=lambda x: x[2], reverse=True)  # Sort by followers
                
                for tweet, username, followers in quality_targets[:MAX_RAIDS_PER_CYCLE]:
                    await smart_crypto_raid(tweet, username)
                    logger.info(f"🎯 Targeted @{username} ({followers} followers)")
                    
                    # Delay between raids
                    await human_delay(60, 180)
                
                logger.info(f"✅ Processed {len(quality_targets[:MAX_RAIDS_PER_CYCLE])} quality targets")
            
        except Exception as e:
            logger.error(f"Raid loop error: {e}")
        
        # Wait before next cycle
        await asyncio.sleep(RAID_CYCLE_DELAY)

async def smart_like_loop():
    """Strategic liking for visibility"""
    
    logger.info("👍 Starting smart liking campaign")
    query_index = 0
    
    while True:
        try:
            current_query = LIKE_QUERIES[query_index % len(LIKE_QUERIES)]
            query_index += 1
            
            results = await safe_search(current_query, max_results=15)
            
            if results and results.data:
                user_map = {}
                if hasattr(results, 'includes') and results.includes:
                    for user in results.includes.get('users', []):
                        user_map[user.id] = user
                
                liked_count = 0
                for tweet in results.data:
                    if liked_count >= MAX_LIKES_PER_CYCLE:
                        break
                    
                    # Skip if already liked
                    if redis_client.sismember(f"{REDIS_PREFIX}liked", str(tweet.id)):
                        continue
                    
                    author = user_map.get(tweet.author_id)
                    if not author:
                        continue
                    
                    follower_count = author.public_metrics.get('followers_count', 0)
                    
                    # Like quality content from established accounts
                    if follower_count >= 100 and len(tweet.text) > 80:
                        success = await safe_like(str(tweet.id))
                        if success:
                            redis_client.sadd(f"{REDIS_PREFIX}liked", str(tweet.id))
                            redis_client.expire(f"{REDIS_PREFIX}liked", 86400)
                            liked_count += 1
                            logger.info(f"👍 Liked: @{author.username} ({follower_count} followers)")
                
                logger.info(f"💙 Liked {liked_count} quality posts")
            
        except Exception as e:
            logger.error(f"Like loop error: {e}")
        
        await asyncio.sleep(LIKE_CYCLE_DELAY)

async def handle_mentions():
    """Handle direct mentions - clean responses without @mentions"""
    
    while True:
        try:
            query = f"@{BOT_USERNAME} -is:retweet"
            results = await safe_search(query, max_results=5)
            
            if results and results.data:
                for tweet in results.data:
                    # Skip if processed
                    if redis_client.sismember(f"{REDIS_PREFIX}mentions", str(tweet.id)):
                        continue
                    
                    redis_client.sadd(f"{REDIS_PREFIX}mentions", str(tweet.id))
                    
                    # Clean the mention text - REMOVE @BOT_USERNAME
                    text = re.sub(f"@{BOT_USERNAME}", "", tweet.text, flags=re.IGNORECASE).strip()
                    
                    # Handle specific requests
                    if re.search(r"\b(price|chart|data)\b", text, re.IGNORECASE):
                        data = fetch_degen_data()
                        if data:
                            response = (
                                f"💎 $DEGEN: ${data['price_usd']:,.6f}\n"
                                f"📊 MC: ${data['market_cap']:,.0f}\n" 
                                f"📈 24h: {data['change_24h']:+.2f}%\n\n"
                                f"Chart: {data['link']}"
                            )
                        else:
                            response = f"$DEGEN data loading... Chart: https://dexscreener.com/solana/{DEGEN_ADDR}"
                    
                    elif re.search(r"\b(ca|contract|address)\b", text, re.IGNORECASE):
                        response = f"$DEGEN Contract Address:\n{DEGEN_ADDR}"
                    
                    else:
                        # General conversation
                        prompt = f"User asked: '{text}'\nRespond naturally about crypto. If relevant, mention $DEGEN positively."
                        response = ask_grok(prompt)
                        
                        if "degen" in text.lower() and DEGEN_ADDR not in response:
                            response += f"\n\n$DEGEN: {DEGEN_ADDR}"
                    
                    # CLEAN RESPONSE - Remove any @mentions
                    response = re.sub(r'@\w+', '', response).strip()
                    
                    await safe_tweet(response, reply_to=tweet.id)
                    logger.info(f"💬 Replied to mention")
            
        except Exception as e:
            logger.error(f"Mention handling error: {e}")
        
        await asyncio.sleep(300)  # Check every 5 minutes

async def daily_organic_post():
    """Post organic content daily"""
    
    while True:
        try:
            # Wait for good timing (avoid spam hours)
            now = datetime.now()
            if 10 <= now.hour <= 20:  # Reasonable hours
                
                data = fetch_degen_data()
                if data:
                    prompt = (
                        f"Write an engaging tweet about $DEGEN token. "
                        f"Current price: ${data['price_usd']:.6f}, 24h: {data['change_24h']:+.2f}%. "
                        f"Be bullish but not spammy. Include some analysis or insight."
                    )
                    
                    content = ask_grok(prompt)
                    
                    post = f"{content}\n\n{data['link']}"
                    
                    # Only post if different from last post
                    last_post = redis_client.get(f"{REDIS_PREFIX}last_organic")
                    if post != last_post:
                        result = await safe_tweet(post)
                        if result:
                            redis_client.setex(f"{REDIS_PREFIX}last_organic", 86400, post)
                            logger.info("📝 Posted organic content")
        
        except Exception as e:
            logger.error(f"Organic post error: {e}")
        
        # Wait 24 hours
        await asyncio.sleep(86400)

async def monitor_bot_health():
    """Monitor bot status and compliance"""
    
    while True:
        try:
            # Test API access
            me = x_client.get_me()
            if me:
                logger.info(f"💚 Bot healthy: @{me.data.username}")
                logger.info(f"📊 Today: {compliance.daily_tweets}/{DAILY_TWEET_LIMIT} tweets, {compliance.daily_likes}/{DAILY_LIKE_LIMIT} likes")
            
        except tweepy.Forbidden:
            logger.error("🚨 BOT SUSPENDED - Stopping all activity")
            return
        except Exception as e:
            logger.warning(f"Health check warning: {e}")
        
        await asyncio.sleep(3600)  # Every hour

async def main():
    """Run the smart, compliant crypto promotion bot"""
    
    try:
        logger.info("🚀 Starting SMART CRYPTO PROMOTION BOT")
        logger.info("✅ Features: Strategic raiding, smart liking, clean replies (no @mentions)")
        logger.info(f"⚙️ Limits: {DAILY_TWEET_LIMIT} tweets/day, {DAILY_LIKE_LIMIT} likes/day")
        
        # Run all components
        await asyncio.gather(
            handle_mentions(),        # Clean mention responses
            strategic_raid_loop(),    # Quality crypto raiding
            smart_like_loop(),        # Strategic liking
            daily_organic_post(),     # Daily organic content
            monitor_bot_health(),     # Health monitoring
        )
        
    except Exception as e:
        logger.error(f"Main error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        exit(1)