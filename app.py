import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. App Configuration (Wide Mode & Title) ---
st.set_page_config(
    page_title="FinBot - Your Banking Assistant",
    page_icon="🏦",
    layout="centered" 
)

# --- 2. Custom CSS Injection (For Premium UI) ---
st.markdown("""
<style>
    /* Main Background aur Font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Area */
    .css-10trblm {
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* User Message Bubble */
    .user-msg {
        background-color: #2b313e;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        margin-bottom: 10px;
        text-align: right;
        border: 1px solid #4a5568;
    }
    
    /* Bot Message Bubble */
    .bot-msg {
        background-color: #1A202C;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        margin-bottom: 10px;
        text-align: left;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Input Box Styling */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Tags for Bank and Category */
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
        margin-bottom: 8px;
    }
    .bank-tag { background-color: #2C5282; color: white; }
    .cat-tag { background-color: #276749; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 3. Header Section ---
st.markdown("<h1 style='text-align: center; color: #63B3ED;'>🏦 FinBot Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AEC0;'>100% Secure & Offline Banking Intelligence</p>", unsafe_allow_html=True)
st.divider()

# --- 4. Load Model & Database (Cached) ---
@st.cache_resource(show_spinner="Waking up the AI...")
def load_system():
    with open('banking_knowledge_base.pkl', 'rb') as f:
        db = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return db, model

db, model = load_system()

# --- 5. Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your smart banking assistant. You can ask me anything about your bank accounts, loans, or services.", "bank": None, "cat": None}
    ]

# --- 6. Display Chat History ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Custom HTML for User
        st.markdown(f"<div class='user-msg'>👤 <b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        # Custom HTML for Bot
        tags_html = ""
        if msg.get("bank") and msg.get("cat"):
            tags_html = f"""
            <span class='tag bank-tag'>🏦 {msg['bank']}</span>
            <span class='tag cat-tag'>📑 {msg['cat']}</span><br>
            """
        st.markdown(f"<div class='bot-msg'>{tags_html}🤖 <b>FinBot:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

# --- 7. Chat Input Logic ---
if prompt := st.chat_input("Ask anything ... (e.g., What is my credit card limit?)"):
    
    # Save and show user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()  # Rerun to display the new user message before processing the bot response

# --- 8. Generate Bot Response ---
if st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.spinner("Searching ..."):
        # Vector Search
        query_vector = model.encode([user_query])
        similarities = cosine_similarity(query_vector, db['vectors'])[0]
        best_match_idx = np.argmax(similarities)
        match_score = similarities[best_match_idx]

        if match_score < 0.45:  # Slightly relaxed threshold
            bot_reply = "Sorry, I don't have the exact answer to that question. Please ask in a clearer way."
            bank_val, cat_val = None, None
        else:
            bot_reply = db['answers'][best_match_idx]
            bank_val = db['banks'][best_match_idx]
            cat_val = db['categories'][best_match_idx]

    # Save bot message and refresh
    st.session_state.messages.append({
        "role": "assistant", 
        "content": bot_reply,
        "bank": bank_val,
        "cat": cat_val
    })
    st.rerun()