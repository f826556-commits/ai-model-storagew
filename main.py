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

MODEL_PATH = "smollm2-135m.gguf"
MODEL_URL = "https://github.com/f826556-commits/-/releases/download/v1.0/SmolLM2-135M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("हल्का दिमाग (135M Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

# n_ctx को 128 से बढ़ाकर 256 कर दिया है ताकि उसे सोचने की जगह मिले
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=256,          
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
        # मैनुअल <|im_start|> को हटाकर सुरक्षित Chat API का इस्तेमाल
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Reply with a single extremely short answer. For math, just give the number."},
                {"role": "user", "content": query.prompt}
            ],
            max_tokens=20
        )
        
        reply = output["choices"][0]["message"]["content"].strip()
        background_tasks.add_task(cleanup_memory)
        
        return {"reply": reply}
        
    except Exception as e:
        background_tasks.add_task(cleanup_memory)
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "100MB वाला हल्का और क्रैश-प्रूफ AI इंजन 100% रेडी है!"}
    
