from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os
import urllib.request
from deep_translator import GoogleTranslator

app = FastAPI()

# आपका मॉडल का नाम और आपका डायरेक्ट GitHub लिंक
MODEL_PATH = "smollm2.gguf"
MODEL_URL = "https://github.com/f826556-commits/ai-model-storagew/releases/download/v1.0/SmolLM2-360M-Instruct-Q4_K_M.gguf"

# सर्वर चालू होते ही यह चेक करेगा कि मॉडल है या नहीं, नहीं है तो GitHub से डाउनलोड करेगा
if not os.path.exists(MODEL_PATH):
    print("आपका दिमाग (Model) डाउनलोड हो रहा है... कृपया प्रतीक्षा करें...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("डाउनलोड पूरा हुआ!")

# 512 MB RAM के लिए सुरक्षित सेटिंग्स (ताकि सर्वर क्रैश न हो)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=256,       # याददाश्त लिमिट
    n_threads=2,     # Render CPU लिमिट
    n_gpu_layers=0   # सिर्फ CPU का इस्तेमाल
)

class Query(BaseModel):
    prompt: str

@app.post("/ask")
def ask_model(query: Query):
    try:
        # 1. यूज़र का हिंदी सवाल इंग्लिश में ट्रांसलेट करें
        english_prompt = GoogleTranslator(source='hi', target='en').translate(query.prompt)
        
        # 2. SmolLM2 का सवाल पूछने का फॉर्मेट
        formatted_prompt = f"<|im_start|>user\n{english_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 3. मॉडल से इंग्लिश में जवाब लें
        output = llm(
            formatted_prompt,
            max_tokens=150,      # रैम बचाने के लिए छोटा जवाब
            stop=["<|im_end|>"],
            echo=False
        )
        english_reply = output["choices"][0]["text"].strip()
        
        # 4. मॉडल के इंग्लिश जवाब को वापस हिंदी में ट्रांसलेट करें
        hindi_reply = GoogleTranslator(source='en', target='hi').translate(english_reply)
        
        return {"reply": hindi_reply}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"status": "आपका खुद का दिमाग 100% काम कर रहा है!"}
  z
