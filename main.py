import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional, List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

class UserInput(BaseModel):
    message: str
    image: Optional[str] = None


chat_history: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful shopping assistant that can analyze product images and answer questions about them."}
]

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(data: UserInput):
    print(f"User asking: {data.message}")
    

    user_content = [{"type": "text", "text": data.message}]
    
    if data.image:
        print(" (Image detected! Attaching for Nvidia...)")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data.image}
        })

    
    chat_history.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            messages=chat_history,
        )
        
        bot_reply = response.choices[0].message.content
        

        chat_history.append({"role": "assistant", "content": bot_reply})
        
        return {"reply": bot_reply}

    except Exception as e:
        print(f"Error calling AI: {e}")
        return {"reply": "I am having trouble seeing that image right now. Please try again or upload a smaller image."}
