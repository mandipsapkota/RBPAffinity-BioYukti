import torch
import torch.nn as nn
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertModel, AutoTokenizer

# ==========================================
# 1. MODEL ARCHITECTURE (Matches Notebook)
# ==========================================
class RBPDualHeadModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(RBPDualHeadModel, self).__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),
            nn.ReLU() # Note: model uses ReLU at the end
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Using the [CLS] token (first token)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        
        class_logits = self.classification_head(last_hidden_state)
        reg_output = self.regression_head(last_hidden_state)
        
        return class_logits, reg_output

# ==========================================
# 2. FASTAPI APP SETUP
# ==========================================
app = FastAPI(title="TDP-43 Binding Prediction API")

# Allow all hosts (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model/tokenizer objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

# Local Path Configuration
MODEL_DIR = "./tdp43_2.0"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.pth")

@app.on_event("startup")
def load_assets():
    global model, tokenizer
    print(f"Loading assets from {MODEL_DIR}...")
    
    try:
        # Load tokenizer from local files
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        # Initialize model architecture
        model = RBPDualHeadModel()
        
        # Load weights
        if os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print(f"Successfully loaded weights from {WEIGHTS_PATH}")
        else:
            print(f"Warning: {WEIGHTS_PATH} not found. Running with uninitialized weights.")
            
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error during startup: {e}")

# ==========================================
# 3. PREPROCESSING & API ENDPOINTS
# ==========================================
class PredictionRequest(BaseModel):
    sequence: str

class PredictionResponse(BaseModel):
    classification: int
    label: str
    confidence: float
    regression_score: float

def preprocess(seq: str):
    # 1. RNA to DNA conversion
    seq = seq.upper().replace("U", "T")
    # 2. 5-mer transformation (crucial for model performance)
    if len(seq) < 5: return seq
    return " ".join(seq[i:i+2] for i in range(len(seq)-1))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.sequence:
        raise HTTPException(status_code=400, detail="No sequence provided")

    # Preprocess
    formatted_seq = preprocess(request.sequence)
    
    # Tokenize
    inputs = tokenizer(
        formatted_seq, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)

    # Inference
    with torch.no_grad():
        logits, reg_val = model(inputs['input_ids'], inputs['attention_mask'])
        
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    return PredictionResponse(
        classification=pred_class,
        label="Binds" if pred_class == 1 else "Does Not Bind",
        confidence=round(probs[0][pred_class].item(), 4),
        regression_score=round(reg_val.item(), 4)
    )

@app.get("/health")
def health():
    return {"status": "online", "device": str(device)}

if __name__ == "__main__":
    # host="0.0.0.0" allows external network access
    uvicorn.run(app, host="0.0.0.0", port=8000)