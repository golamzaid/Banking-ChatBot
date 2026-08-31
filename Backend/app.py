from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import gc
from sentence_transformers import SentenceTransformer
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

# Global variables (initially None to save startup RAM)
db = None
model = None

def get_system():
    global db, model
    if db is None:
        print("Loading Knowledge Base...")
        with open('banking_knowledge_base.pkl', 'rb') as f:
            db = pickle.load(f)
    if model is None:
        print("Loading Lightweight Model...")
        # CPU optimization ke sath model load karein taaki RAM kam khaye
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return db, model

class ChatQuery(BaseModel):
    prompt: str

@app.post("/api/chat")
async def predict_response(query: ChatQuery):
    user_query = query.prompt.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # Get optimized system instance
    database, ai_model = get_system()
    
    # Vector Search Logic
    query_vector = ai_model.encode([user_query])
    similarities = cosine_similarity(query_vector, database['vectors'])[0]
    best_match_idx = np.argmax(similarities)
    match_score = similarities[best_match_idx]
    
    # Memory cleanup
    gc.collect()
    
    if match_score < 0.45:
        return {
            "reply": "Sorry, I don't have the exact answer to that question. Please ask in a clearer way.",
            "bank": None,
            "category": None,
            "score": float(match_score)
        }
    
    return {
        "reply": database['answers'][best_match_idx],
        "bank": database['banks'][best_match_idx],
        "category": database['categories'][best_match_idx],
        "score": float(match_score)
    }

@app.get("/")
def home():
    return {"status": "FinBot Optimized Backend is running!"}