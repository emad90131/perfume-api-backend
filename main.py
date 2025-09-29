import os
import sqlite3
import datetime
import pandas as pd
from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq
from starlette.status import HTTP_403_FORBIDDEN

# Load environment variables
load_dotenv()

# --- Database Setup ---
DB_FILE = "chat_history.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     timestamp TEXT,
     username TEXT,
     query TEXT,
     response TEXT,
     context TEXT)
''')
conn.commit()

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    query: str
    history: List[ChatMessage] = []
    context: Optional[str] = None
    username: Optional[str] = "guest"

class RecommendationResponse(BaseModel):
    recommendation: str
    context: str

# --- FastAPI App and CORS ---
app = FastAPI(title="Perfume Store AI Assistant")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Client and Data Loading ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant" 

try:
    df_perfumes = pd.read_csv("perfumes.csv")
    df_perfumes.fillna('', inplace=True) 
except FileNotFoundError:
    df_perfumes = None

# --- API Endpoint ---
@app.post("/get-recommendation", response_model=RecommendationResponse)
def get_perfume_recommendation(user_query: UserQuery):
    if not os.getenv("GROQ_API_KEY") or df_perfumes is None:
        return {"recommendation": "Internal server configuration error.", "context": ""}

    perfumes_context = ""
    if user_query.context:
        perfumes_context = user_query.context
    else:
        # Smart search logic
        query_lower = user_query.query.lower()
        final_list_to_sample = df_perfumes
        num_samples = min(10, len(final_list_to_sample))
        top_perfumes = final_list_to_sample.sample(n=num_samples) if num_samples > 0 else pd.DataFrame()
        if not top_perfumes.empty:
            perfumes_context = top_perfumes.to_json(orient="records", force_ascii=False)

    system_prompt = "You are an AI assistant for a perfume store..." # Your full system prompt
    
    messages = [{'role': 'system', 'content': f"{system_prompt}\n\nPERFUME LIST: {perfumes_context}"}]
    for msg in user_query.history:
        messages.append({'role': msg.role, 'content': msg.content})
    
    final_user_prompt = f"Based on the list, answer: \"{user_query.query}\"..." # Your full format prompt
    messages.append({'role': 'user', 'content': final_user_prompt})
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=MODEL_NAME, temperature=0.7, max_tokens=500,
        )
        response_text = chat_completion.choices[0].message.content
        
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO conversations (timestamp, username, query, response, context) VALUES (?, ?, ?, ?, ?)",
            (timestamp, user_query.username, user_query.query, response_text, perfumes_context)
        )
        conn.commit()
        
        return {"recommendation": response_text, "context": perfumes_context}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"recommendation": f"Sorry, an error occurred: {e}", "context": perfumes_context}

# --- Health Check Endpoint ---
@app.get("/")
def health_check():
    return {"status": "ok"}
