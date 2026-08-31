import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# 1. Apna Dataset Load Karein (Agar CSV hai toh pd.read_csv use karein)
# Dhyan dein: 'your_dataset.csv' ko apni actual file ke naam se replace karein
df = pd.read_csv(r"C:\Users\golam\OneDrive\Desktop\CODES\PROJECTS\Banking ChatBot - org\Datasets\master_banking_kb.csv")
# 2. Data Clean Karein (Missing rows hatayein taaki misalignment na ho)
# Assumed columns: 'Question', 'Answer', 'Bank', 'Category'
df = df.dropna(subset=['Question', 'Answer'])
df = df.reset_index(drop=True)  # YEH SABSE ZAROORI STEP HAI MISALIGNMENT ROKNE KE LIYE!

print(f"Total valid Q&A pairs found: {len(df)}")

# 3. Model Load Karein
print("Loading Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Embeddings Generate Karein
print("Generating vectors... (Isme thoda time lag sakta hai)")
questions = df['Question'].astype(str).tolist()
vectors = model.encode(questions, show_progress_bar=True)

# 5. Extra Columns Handle Karein (Agar bank/category column nahi hai toh default set karein)
banks = df['Bank'].tolist() if 'Bank' in df.columns else ["General"] * len(df)
categories = df['Category'].tolist() if 'Category' in df.columns else ["General"] * len(df)

# 6. Naya Pickle File Save Karein
knowledge_base = {
    'vectors': vectors,
    'answers': df['Answer'].tolist(),
    'banks': banks,
    'categories': categories
}

with open('banking_knowledge_base.pkl', 'wb') as f:
    pickle.dump(knowledge_base, f)

print("✅ Success: 'banking_knowledge_base.pkl' properly generated and aligned!")