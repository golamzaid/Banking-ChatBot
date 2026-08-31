from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="FinBot API", version="1.0")

#  CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Load Model & Database Once on Startup 
try:
    with open('banking_knowledge_base.pkl', 'rb') as f:
        db = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading model or database: {e}")

# Request Body Schema
class ChatQuery(BaseModel):
    prompt: str

# Prediction / Chat Endpoint 
@app.post("/api/chat")
async def predict_response(query: ChatQuery):
    user_query = query.prompt.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # Vector Search Logic
    query_vector = model.encode([user_query])
    similarities = cosine_similarity(query_vector, db['vectors'])[0]
    best_match_idx = np.argmax(similarities)
    match_score = similarities[best_match_idx]
    
    if match_score < 0.45:
        return {
            "reply": "Sorry, I don't have the exact answer to that question. Please ask in a clearer way.",
            "bank": None,
            "category": None,
            "score": float(match_score)
        }
    
    return {
        "reply": db['answers'][best_match_idx],
        "bank": db['banks'][best_match_idx],
        "category": db['categories'][best_match_idx],
        "score": float(match_score)
    }

@app.get("/")
def home():
    return {"status": "FinBot Backend is running successfully!"}