import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import datetime

# (باقي الإعدادات الأولية وقاعدة البيانات تبقى كما هي)
# ...
DB_FILE = "chat_history.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, username TEXT, query TEXT, response TEXT, context TEXT)''')
conn.commit()
load_dotenv()
app = FastAPI(title="متجر العطور - مساعد الذكاء الاصطناعي")
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant" 
try:
    df_perfumes = pd.read_csv("perfumes.csv")
    df_perfumes.fillna('', inplace=True) 
except FileNotFoundError:
    df_perfumes = None

class ChatMessage(BaseModel):
    role: str
    content: str
class UserQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)
    context: Optional[str] = None
    username: Optional[str] = "guest"
class RecommendationResponse(BaseModel):
    recommendation: str
    context: str

# --- نقطة النهاية ---
@app.post("/get-recommendation", response_model=RecommendationResponse)
def get_perfume_recommendation(user_query: UserQuery):
    if not os.getenv("GROQ_API_KEY") or df_perfumes is None:
        return {"recommendation": "خطأ في الإعدادات الداخلية.", "context": ""}

    perfumes_context = ""
    # (منطق الذاكرة والبحث الذكي يبقى كما هو)
    if user_query.context:
        perfumes_context = user_query.context
    else:
        # ... (Your smart search logic) ...
        final_list_to_sample = df_perfumes
        num_samples = min(10, len(final_list_to_sample))
        top_perfumes = final_list_to_sample.sample(n=num_samples) if num_samples > 0 else pd.DataFrame()
        if not top_perfumes.empty:
            perfumes_context = top_perfumes.to_json(orient="records", force_ascii=False)

    system_prompt = f"You are an AI assistant for a perfume store... PERFUME LIST: {perfumes_context}"
    
    messages = [{'role': 'system', 'content': system_prompt}]
    for msg in user_query.history:
        messages.append({'role': msg.role, 'content': msg.content})
    
    # ===> منطق تحديد النية المحدث <===
    query_lower = user_query.query.lower()
    information_keywords = ["مكونات", "معلومات عن", "ما هو", "حدثني عن"]
    greeting_keywords = ["مرحبا", "السلام عليكم", "صباح الخير", "مساء الخير", "كيف حالك", "شكرا", "تحية", "اهلا", "أهلا"]

    is_greeting = any(keyword in query_lower for keyword in greeting_keywords)
    is_information_request = any(keyword in query_lower for keyword in information_keywords)

    if is_greeting and len(user_query.query.split()) <= 3:
        # 1. موجه خاص بالتحية والمحادثات العامة
        final_user_prompt = f"""The user has just said: "{user_query.query}".
        
        **CRITICAL INSTRUCTION:** This is a general greeting or a short conversational phrase.
        Respond naturally and politely in Arabic. DO NOT recommend a perfume.
        Simply greet them back and ask how you can help them find a perfume.
        """
    elif is_information_request:
        # 2. موجه خاص بطلبات المعلومات
        final_user_prompt = f"""Based on the JSON list and conversation history, answer this specific question: "{user_query.query}"

        **CRITICAL INSTRUCTION:** Answer ONLY the specific question. DO NOT suggest another perfume. Your response MUST be in Arabic.
        """
    else:
        # 3. الموجه الافتراضي لطلبات التوصية
        final_user_prompt = f"""Based on the JSON list provided, answer this customer request: "{user_query.query}"

        **CRITICAL INSTRUCTIONS:**
        1.  **Primary Choice:** Select the single best perfume from the list that matches the request.
        2.  **Alternative Choice:** Select a second, different perfume from the list that is also a good alternative.
        3.  **Format (MANDATORY):** Follow this exact format... (your full recommendation format)
        """
    # =========================================================
        
    messages.append({'role': 'user', 'content': final_user_prompt})
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=MODEL_NAME, temperature=0.6, max_tokens=500,
        )
        response_text = chat_completion.choices[0].message.content
        
        # (منطق حفظ المحادثة يبقى كما هو)
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