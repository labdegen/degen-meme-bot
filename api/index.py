# api/index.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Degen Meme Bot is live. Mention me with a $TOKEN!"}
