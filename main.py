from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. आपका नया और हल्का मॉडल (135M)
MODEL_PATH = "smollm2-135m.gguf"
MODEL_URL = "https://github.com/f826556-commits/-/releases/download/v1.0/SmolLM2-135M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("हल्का दिमाग (135M Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

# 2. EXTREME RAM SAVING सेटिंग्स (512MB RAM के लिए)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,          
    n_threads=1,        
    n_batch=8,          
    n_gpu_layers=0,
    use_mmap=True,      
    use_mlock=False     
)

class Query(BaseModel):
    prompt: str

def cleanup_memory():
    gc.collect()

@app.post("/ask")
def ask_model(query: Query, background_tasks: BackgroundTasks):
    try:
        system_prompt = "Reply with a single extremely short answer. For math, just give the number."
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        output = llm(
            formatted_prompt,
            max_tokens=20,       
            stop=["<|im_end|>"],
            echo=False
        )
        
        reply = output["choices"][0]["text"].strip()
        background_tasks.add_task(cleanup_memory)
        
        return {"reply": reply}
        
    except Exception as e:
        background_tasks.add_task(cleanup_memory)
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "100MB वाला हल्का और क्रैश-प्रूफ AI इंजन 100% रेडी है!"}
    
