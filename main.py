from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request
import gc

# 1. FastAPI सर्वर चालू करना
app = FastAPI()

# 2. CORS सेटिंग्स (ताकि Netlify का फ्रंटएंड इससे बात कर सके)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. आपका अपना पर्सनल मॉडल (SmolLM2)
MODEL_PATH = "smollm2.gguf"
MODEL_URL = "https://github.com/f826556-commits/ai-model-storagew/releases/download/v1.0/SmolLM2-360M-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("दिमाग (Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

# 4. EXTREME RAM SAVING सेटिंग्स (512MB RAM के लिए ब्रह्मास्त्र)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,          # याददाश्त छोटी रखी है (ताकि रैम न भरे)
    n_threads=1,        # 1 धागे से शांति से सोचेगा
    n_batch=8,          # 8-8 शब्दों के छोटे टुकड़ों में प्रोसेस करेगा (नो क्रैश)
    n_gpu_layers=0,
    use_mmap=True,      # मॉडल को रैम के बजाय हार्ड डिस्क से पढ़ेगा
    use_mlock=False     # रैम को लॉक होने से रोकेगा
)

# यूज़र के सवाल का स्ट्रक्चर
class Query(BaseModel):
    prompt: str

# 5. स्मार्ट कचरा सफाई फंक्शन (Background Cleaner)
def cleanup_memory():
    gc.collect()

# 6. सवाल पूछने वाला मेन API रूट
@app.post("/ask")
def ask_model(query: Query, background_tasks: BackgroundTasks):
    try:
        # AI को सख्त हिदायत (System Prompt)
        system_prompt = "Reply with a single extremely short answer. For math, just give the number."
        
        # मॉडल के समझने लायक फॉर्मेट
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 7. मॉडल का सोचना शुरू (सिर्फ 20 शब्दों की लिमिट के साथ)
        output = llm(
            formatted_prompt,
            max_tokens=20,       
            stop=["<|im_end|>"],
            echo=False
        )
        
        reply = output["choices"][0]["text"].strip()
        
        # 8. जवाब भेजते ही बैकग्राउंड में कचरा साफ करने का आदेश
        background_tasks.add_task(cleanup_memory)
        
        return {"reply": reply}
        
    except Exception as e:
        # अगर कोई एरर आए, तब भी रैम साफ करेगा
        background_tasks.add_task(cleanup_memory)
        return {"error": str(e)}

# 9. हेल्थ चेक रूट
@app.get("/")
def read_root():
    return {"status": "पर्सनल AI इंजन (Pro Version) 100% रेडी और लाइव है!"}
    
