import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

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

chat_history = [
    {"role": "system", "content": "You are a helpful expert Shopping Assistant. You remember context from previous messages."}
]

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(data: UserInput):
    full_user_message = data.message
    
    if data.image:
        try:
            vision_response = client.chat.completions.create(
                model="qwen/qwen-2-vl-7b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this product in detail for a shopper."},
                            {"type": "image_url", "image_url": {"url": data.image}}
                        ]
                    }
                ],
            )
            description = vision_response.choices[0].message.content
            full_user_message += f"\n[User uploaded an image of: {description}]"
        except Exception:
            full_user_message += "\n[User uploaded an image, but I couldn't see it.]"

    chat_history.append({"role": "user", "content": full_user_message})

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=chat_history,
    )
    
    bot_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": bot_reply})
    
    return {"reply": bot_reply}