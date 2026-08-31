from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="FinBot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Database safely on startup (Uses very low RAM)
try:
    with open('banking_knowledge_base.pkl', 'rb') as f:
        db = pickle.load(f)
    print("Database loaded successfully!")
except Exception as e:
    print(f"Error loading database: {e}")

class ChatQuery(BaseModel):
    prompt: str

@app.post("/api/chat")
async def predict_response(query: ChatQuery):
    user_query = query.prompt.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # Simple keyword/matching or if vectors exist, use them
    # Note: If your .pkl only has vectors, we match them. 
    # To keep it lightweight without sentence-transformers, we can do a direct text search or basic matching:
    
    # Basic keyword search fallback to save 100% RAM and avoid heavy ML crashes
    user_words = set(user_query.lower().split())
    best_match_idx = 0
    max_matches = -1
    
    for idx, ans in enumerate(db['answers']):
        # Simple text overlap scoring
        ans_words = set(ans.lower().split())
        common = len(user_words.intersection(ans_words))
        if common > max_matches:
            max_matches = common
            best_match_idx = idx

    if max_matches <= 0 and len(db['answers']) > 0:
        return {
            "reply": "Sorry, I don't have the exact answer to that question. Please ask in a clearer way.",
            "bank": None,
            "category": None
        }

    return {
        "reply": db['answers'][best_match_idx],
        "bank": db['banks'][best_match_idx],
        "category": db['categories'][best_match_idx]
    }

@app.get("/")
def home():
    return {"status": "FinBot Lightweight Backend is running!"}