from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request

app = FastAPI()

# CORS सेटिंग ताकि आपका HTML पेज इससे जुड़ सके
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# आपका मॉडल और गिटहब लिंक
MODEL_PATH = "smollm2.gguf"
MODEL_URL = "https://github.com/f826556-commits/ai-model-storagew/releases/download/v1.0/SmolLM2-360M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("दिमाग (Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

# ---- रैम बचाने की एक्सट्रीम सेटिंग ----
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,       # याददाश्त को बहुत कम कर दिया है (128) ताकि रैम न भरे
    n_threads=1,     # सिर्फ 1 धागे (Thread) से शांति से सोचेगा
    n_batch=128,     # डेटा को छोटे-छोटे 128 के पैकेट में प्रोसेस करेगा
    n_gpu_layers=0
)

class Query(BaseModel):
    prompt: str

@app.post("/ask")
def ask_model(query: Query):
    try:
        # सवाल का फॉर्मेट
        formatted_prompt = f"<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # मॉडल सोचना शुरू करेगा
        output = llm(
            formatted_prompt,
            max_tokens=100,      # आपका आईडिया: सिर्फ 100 टोकन (शब्द) तक ही सोचेगा ताकि क्रैश न हो
            stop=["<|im_end|>"],
            echo=False
        )
        
        reply = output["choices"][0]["text"].strip()
        return {"reply": reply}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "प्योर दिमाग (Extreme RAM Saving) 100% काम कर रहा है!"}
    
