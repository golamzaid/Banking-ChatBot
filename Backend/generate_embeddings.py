import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


df = pd.read_csv(r"C:\Users\golam\OneDrive\Desktop\CODES\PROJECTS\Banking ChatBot - org\Datasets\master_banking_kb.csv")

df = df.dropna(subset=['Question', 'Answer'])
df = df.reset_index(drop=True) 

print(f"Total valid Q&A pairs found: {len(df)}")


print("Loading Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embeddings 
print("Generating vectors... (Isme thoda time lag sakta hai)")
questions = df['Question'].astype(str).tolist()
vectors = model.encode(questions, show_progress_bar=True)

banks = df['Bank'].tolist() if 'Bank' in df.columns else ["General"] * len(df)
categories = df['Category'].tolist() if 'Category' in df.columns else ["General"] * len(df)


knowledge_base = {
    'vectors': vectors,
    'answers': df['Answer'].tolist(),
    'banks': banks,
    'categories': categories
}

with open('banking_knowledge_base.pkl', 'wb') as f:
    pickle.dump(knowledge_base, f)

print("✅ Success: 'banking_knowledge_base.pkl' properly generated and aligned!")