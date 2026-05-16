# This code is designed to create a knowledge base for a banking assistant application. 
# It loads a dataset of FAQs, processes the questions into vector representations using a pre-trained model, 
# and saves the processed data into a file that can be used in an Android app or UI.
# The code uses the `pandas` library for data manipulation, `sentence_transformers` for generating question vectors, and `pickle` for saving the processed data.


import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import time

# 1. Dataset Load 
print("Dataset is loading...")
df = pd.read_csv(r'Database\banking_assistant_50k_faqs.csv')

# empty string for NaN values
df.fillna('', inplace=True)

# retrieving data to different lists
ref_ids = df['Reference_ID'].tolist()
banks = df['Bank'].tolist()
categories = df['Category'].tolist()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# 2. Vector Model Load (Offline & Lightweight)
print("Model is loading (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. 50k Questions converted to Vectors (Numbers)
print(f"Total {len(questions)} questions are being converted to vectors (it may take a while)...")
start_time = time.time()
question_vectors = model.encode(questions, show_progress_bar=True)
print(f"Vectors generated! Time taken: {round(time.time() - start_time, 2)} seconds")

# 4. create database or dictionary for android app.
data_to_save = {
    'vectors': question_vectors,
    'questions': questions,
    'answers': answers,
    'banks': banks,
    'categories': categories,
    'ref_ids': ref_ids
}

# 5. save the data to a file using pickle
file_name = 'banking_knowledge_base.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"✅ Success! your on-device database '{file_name}' is ready.")
print("you can now use it in your Android app or UI!")