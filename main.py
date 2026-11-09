from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request
from deep_translator import GoogleTranslator

app = FastAPI()

# यह कोड CORS एरर को खत्म करेगा
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "smollm2.gguf"
MODEL_URL = "https://github.com/f826556-commits/ai-model-storagew/releases/download/v1.0/SmolLM2-360M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("आपका दिमाग (Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=256,
    n_threads=2,
    n_gpu_layers=0
)

class Query(BaseModel):
    prompt: str

@app.post("/ask")
def ask_model(query: Query):
    try:
        english_prompt = GoogleTranslator(source='hi', target='en').translate(query.prompt)
        formatted_prompt = f"<|im_start|>user\n{english_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        output = llm(
            formatted_prompt,
            max_tokens=150,
            stop=["<|im_end|>"],
            echo=False
        )
        english_reply = output["choices"][0]["text"].strip()
        hindi_reply = GoogleTranslator(source='en', target='hi').translate(english_reply)
        
        return {"reply": hindi_reply}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "आपका खुद का दिमाग 100% काम कर रहा है!"}
    
