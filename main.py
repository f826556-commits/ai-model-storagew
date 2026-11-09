from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "smollm2.gguf"
MODEL_URL = "https://github.com/f826556-commits/ai-model-storagew/releases/download/v1.0/SmolLM2-360M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("दिमाग डाउनलोड हो रहा है...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# एक्सट्रीम रैम सेविंग सेटिंग्स
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,       
    n_threads=1,     
    n_batch=64,      # डेटा पैकेट को 128 से घटाकर 64 कर दिया है ताकि प्रोसेसिंग बहुत हल्की हो जाए
    n_gpu_layers=0
)

class Query(BaseModel):
    prompt: str

@app.post("/ask")
def ask_model(query: Query):
    try:
        # AI का दिमाग सेट करने वाली हिदायत (कि उसे कैसे बर्ताव करना है)
        system_prompt = "You are a friend chatting on WhatsApp. Reply with extremely short, natural, and realistic answers. Never write more than one sentence."
        
        # सवाल को मॉडल के समझने लायक फॉर्मेट में डालना
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # मॉडल सोचना शुरू करेगा
        output = llm(
            formatted_prompt,
            max_tokens=30,       # सिर्फ 30 शब्दों की लिमिट (ताकि रैम फुल ही न हो पाए)
            stop=["<|im_end|>"],
            echo=False
        )
        
        reply = output["choices"][0]["text"].strip()
        return {"reply": reply}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "स्मार्ट और छोटा दिमाग 100% काम कर रहा है!"}
    
