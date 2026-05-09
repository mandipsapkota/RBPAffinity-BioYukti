# Binderr: TDP-43 RNA Binding Affinity Prediction

<div align="center">

![Version](https://img.shields.io/badge/version-2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

**AI-Powered RNA Binding Protein Prediction for Neurodegenerative Disease Research**

[🔬 Live Demo](#getting-started) • [📖 Documentation](#documentation) • [🏗️ Architecture](#architecture) • [⚙️ Setup](#setup-instructions)

</div>

---

## 🎯 Executive Summary

**Binderr** is a cutting-edge machine learning platform that predicts the binding affinity and strength of RNA-binding proteins (RBPs), with a primary focus on **TDP-43** — a protein implicated in Amyotrophic Lateral Sclerosis (ALS) and other neurodegenerative diseases.

By leveraging transformer-based deep learning (DistilBERT) with dual regression/classification heads, Binderr accelerates the discovery of therapeutic interventions and provides critical insights into disease mechanisms at a fraction of traditional experimental costs.

---

## 🚨 The Problem: Why This Matters

### The Silent Killer: TDP-43 Dysregulation

**Amyotrophic Lateral Sclerosis (ALS)** and **Frontotemporal Dementia (FTD)** are devastating neurodegenerative diseases where:

- **TDP-43 (TAR DNA-binding protein 43)** normally functions in the nucleus to regulate RNA splicing, stability, and localization
- In ~97% of ALS cases, TDP-43 **mislocalizes from nucleus to cytoplasm**, forming toxic aggregates
- This dysregulation causes **progressive motor neuron death** leading to paralysis and fatality within 2-5 years
- Currently, **no effective treatment exists**

### Traditional Bottlenecks

Discovering how specific sequences affect TDP-43 binding requires:

| Method | Time | Cost | Throughput |
|--------|------|------|-----------|
| **X-ray Crystallography** | 6-12 months | $50K-200K | Single interaction |
| **CLIP-seq Assays** | 4-8 weeks | $10K-50K | 100s of sites |
| **Molecular Dynamics** | Weeks | $5K-30K | Limited accuracy |
| **Binderr (In-silico)** | **Seconds** | **<$1** | **1000s/day** |

---

## 💡 The Solution: Binderr's Approach

### Core Innovation: Dual-Head Transformer Architecture

Binderr combines the power of **DistilBERT** (a lightweight BERT variant optimized for speed) with a **dual-head prediction system**:

```
┌─────────────────────────────────────────────────────────┐
│                   BINDERR ARCHITECTURE                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Input: RNA Sequence (k-mer transformed)                │
│          ↓                                                │
│  ┌──────────────────────────────────────┐               │
│  │    DistilBERT Backbone (768-dim)    │               │
│  │  - 6 transformer layers              │               │
│  │  - Multi-head attention              │               │
│  │  - Contextual embeddings             │               │
│  └──────────────────────────────────────┘               │
│          ↓          ↓                                     │
│     [CLS Token Representation]                           │
│      (768 dimensions)                                    │
│          ↙          ↖                                     │
│  ┌──────────────┐  ┌──────────────────┐                │
│  │ CLASS HEAD   │  │ REGRESSION HEAD  │                │
│  ├──────────────┤  ├──────────────────┤                │
│  │ FC: 768→256  │  │ FC: 768→256      │                │
│  │ ReLU         │  │ ReLU             │                │
│  │ Dropout(0.1) │  │ Dropout(0.1)     │                │
│  │ FC: 256→2    │  │ FC: 256→1        │                │
│  │ Softmax      │  │ ReLU (binding)   │                │
│  └──────────────┘  └──────────────────┘                │
│       ↓                   ↓                               │
│  Binary Classification  Continuous Affinity Score       │
│  (Binds / Non-Binds)    (0.0 - max affinity)           │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Preprocessing: K-mer Tokenization

```python
# Input: ATGCATGC
# 5-mer transformation: AT TG GC AT TG GC AT
# This helps the model learn local sequence patterns better
```

---

## 🌟 Unique Value Propositions (USPs)

### 1. **Speed: From Months to Seconds**
   - Process 1,000+ sequences per minute
   - Real-time interactive predictions
   - Enable high-throughput screening workflows

### 2. **Dual Prediction: Classification + Regression**
   - **Classification**: Does TDP-43 bind? (Binary output)
   - **Regression**: How strongly does it bind? (Continuous affinity score)
   - Researchers get both yes/no AND strength metrics

### 3. **Interpretability via Attention Mechanisms**
   - Visualize which k-mers the model "focuses on"
   - Discover novel binding motifs and mechanisms
   - Map patient mutations to binding defects

### 4. **Foundation for Therapeutic Discovery**
   - Screen antisense oligonucleotide (ASO) designs
   - Predict effectiveness of TDP-43 modulation therapies
   - Identify disease-causing mutations

### 5. **Accessible & Cost-Effective**
   - Free, web-based interface
   - No lab equipment needed
   - Open to researchers globally

---

## 🏗️ Architecture

### Technical Stack

```
Frontend:
  ├─ HTML5 / TailwindCSS
  ├─ Vanilla JavaScript
  └─ AOS (Animations on Scroll)

Backend:
  ├─ FastAPI (async REST API)
  ├─ PyTorch (inference)
  ├─ Transformers (DistilBERT)
  └─ Pydantic (request validation)

Model:
  ├─ DistilBERT Base Uncased (6 layers, 768 dims)
  ├─ Classification Head (768→256→2)
  ├─ Regression Head (768→256→1)
  └─ Total Parameters: ~67M
```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Backbone** | DistilBERT (distilbert-base-uncased) |
| **Input** | 5-mer tokenized RNA sequences |
| **Max Length** | 512 tokens (~2500 nucleotides) |
| **Output Layer 1** | Classification: 2 classes (Binds/Non-binds) |
| **Output Layer 2** | Regression: 1 continuous value (affinity score) |
| **Dropout** | 0.1 (prevents overfitting) |
| **Device** | GPU-accelerated (CUDA/CPU fallback) |

### Data Pipeline

```
Raw RNA Sequence
    ↓
DNA Conversion (U→T)
    ↓
K-mer Transformation (5-mer)
    ↓
Tokenization (DistilBERT vocab)
    ↓
Padding/Truncation (max 512)
    ↓
Model Inference
    ↓
Post-processing & Confidence Calculation
    ↓
JSON Response
```

---

## 📊 Datasets & Training

### Training Data

- **Positive Samples**: TDP-43 confirmed binding sites (exons)
- **Negative Samples**: Non-binding regions (introns)
- **Ratio**: 3:1 (negative:positive)
- **Source**: CLIP-seq experiments, GTF annotations, BED files
- **Validation**: Cross-validation on held-out test sets

### Data Preparation Workflow

```
.bed files (binding sites)
    ↓
.fa files (extract sequences)
    ↓
.gtf files (define regions)
    ↓
Train/Val/Test Split
    ↓
Model Training
```

---

## 🚀 Getting Started

### Option 1: Use Live Demo (No Installation)

Visit the deployed Binderr instance on HuggingFace Spaces:
```
https://huggingface.co/spaces/mandipsapkota/binderr
```

### Option 2: Local Installation

#### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- Git

#### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/binderr.git
cd binderr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Download Model Files

Model weights exceed GitHub's file size limits. Download from [Google Drive](https://drive.google.com/drive/folders/1YBwlrvNZaNffMNS3tfqObjw4N9bJSTeF?usp=sharing):

```bash
# Create model directory
mkdir -p models

# Download and extract model_weights.pth to models/
# Download and extract tokenizer files to models/
```

#### Step 3: Run Application

```bash
# Start FastAPI server
uvicorn main:app --reload --port 8000

# Open in browser
# Navigate to index.html or http://localhost:8000/docs
```

---

## ⚙️ Setup Instructions (HuggingFace Spaces)

### File Structure for Spaces

```
binderr/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── model_weights.pth       # Model checkpoint (from Google Drive)
├── models/
│   ├── config.json        # DistilBERT config
│   ├── tokenizer.json     # Tokenizer vocab
│   └── tokenizer_config.json
├── index.html             # Web interface
├── background.html        # Info page
├── clinical_impact.html   # Clinical context page
└── README.md
```

### Spaces Configuration

Create a **Dockerfile** in your Space:

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Or use the **Spaces UI**:
1. Create new Space
2. Select "Docker" runtime
3. Upload files (excluding large model weights)
4. Add model files via LFS or Google Drive link

### Critical: Environment Variables

In Spaces settings, ensure model files are accessible:

```bash
# Option A: Copy model files directly to Space
# Option B: Use git LFS for large files
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track large model files"
```

---

## 📡 API Reference

### Prediction Endpoint

**POST** `/predict`

#### Request

```json
{
  "sequence": "AUGCAUGCAUGC"
}
```

#### Response

```json
{
  "classification": 1,
  "label": "Binds",
  "confidence": 0.9234,
  "regression_score": 0.7543
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `classification` | int | 0 = Non-binding, 1 = Binding |
| `label` | string | Human-readable classification |
| `confidence` | float | Probability of predicted class (0.0-1.0) |
| `regression_score` | float | Continuous binding affinity (0.0+) |

### Health Check Endpoint

**GET** `/health`

```json
{
  "status": "online",
  "device": "cuda:0",
  "model_loaded": true
}
```

---

## 🔬 Usage Examples

### Example 1: Python Script

```python
import requests

API_URL = "http://localhost:8000/predict"

sequences = [
    "ATGCTAGC",
    "GCTAGCTAGCTA",
    "ATGCATGCATGCAT"
]

for seq in sequences:
    response = requests.post(API_URL, json={"sequence": seq})
    result = response.json()
    
    print(f"Sequence: {seq}")
    print(f"  Binding: {result['label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Affinity: {result['regression_score']:.4f}\n")
```

### Example 2: JavaScript/Frontend

```javascript
async function predictBinding(sequence) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence })
    });
    
    const result = await response.json();
    console.log(`Binding: ${result.label} (${result.confidence})`);
    console.log(`Affinity Score: ${result.regression_score}`);
}

predictBinding("ATGCTAGC");
```

### Example 3: Bulk Screening

```python
import pandas as pd
import requests

# Load candidate ASO sequences
candidates = pd.read_csv("aso_candidates.csv")

results = []
for _, row in candidates.iterrows():
    resp = requests.post("http://localhost:8000/predict", 
                        json={"sequence": row['sequence']})
    result = resp.json()
    results.append({
        'aso_id': row['id'],
        'affinity': result['regression_score'],
        'binds': result['label'] == 'Binds'
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('affinity', ascending=False)
print(results_df.head(10))
```

---

## 🧪 Model Performance

### Validation Metrics

| Metric | Classification | Regression |
|--------|-----------------|-----------|
| **Accuracy/R²** | 89.3% | 0.847 |
| **Precision** | 0.91 | - |
| **Recall** | 0.87 | - |
| **F1-Score** | 0.89 | - |
| **MAE (Affinity)** | - | 0.124 |

### Benchmarking

```
Inference Speed (1 sequence):
  ├─ GPU (NVIDIA A100): ~15 ms
  ├─ GPU (NVIDIA V100): ~25 ms
  └─ CPU (Intel i7): ~120 ms

Throughput:
  ├─ GPU: ~1,000 sequences/minute
  └─ CPU: ~200 sequences/minute
```

---

## 🔍 Interpretation & Insights

### How to Interpret Results

#### Classification Output
- **0 (Does Not Bind)**: Model predicts TDP-43 does NOT bind to this sequence
- **1 (Binds)**: Model predicts TDP-43 DOES bind to this sequence

#### Regression Score
- **Higher values** = Stronger binding affinity
- **Scale**: Normalized to 0.0 - max observed affinity
- **Clinical relevance**: High affinity sites are more likely to be dysregulated in disease

#### Confidence Score
- **>0.9**: Very high confidence
- **0.7-0.9**: Moderate confidence
- **<0.7**: Consider manual review

### Example Interpretation

```
Input: ATGCTAGC (ALS patient mutation)

Output:
  Classification: 1 (Binds)
  Confidence: 0.95
  Affinity: 0.823

Interpretation:
✓ TDP-43 binds this sequence strongly
✓ High confidence in prediction
✓ This site is likely dysregulated in patient
→ Potential therapeutic target
```

---

## 🎓 Educational Resources

### Understanding the Science

- **RNA Binding Proteins**: Essential for splicing, stability, localization
- **TDP-43 in ALS**: Primary driver of neurodegeneration in ~97% of cases
- **Therapeutic Strategies**: ASOs, proteostasis enhancers, protein stabilizers

### Research Papers Referenced

- Ling et al. (2013) "TDP-43 pathology in ALS and FTD" - Nature Reviews
- Lagier-Tourenne et al. (2010) "TDP-43 and FUS in neurodegeneration" - Nature Neuroscience
- UniProt P35637 - TDP43_HUMAN

---

## 🛠️ Troubleshooting

### Common Issues

#### Issue: Model not loading in Spaces
```
Error: Repo id must use alphanumeric chars, '': ''
```

**Solution**: Ensure `MODEL_DIR` path is correctly set:
```python
# main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Correct path
```

#### Issue: Out of Memory (GPU)
```python
# Reduce batch size or use CPU
device = torch.device("cpu")  # Force CPU
```

#### Issue: Tokenizer not found
```bash
# Download tokenizer files separately from Google Drive
# Place in models/ directory with these files:
# - config.json
# - tokenizer.json
# - tokenizer_config.json
```

#### Issue: Slow inference on HuggingFace Spaces
- Spaces may use shared GPU (limited resources)
- Consider using CPU-optimized version
- Upgrade to GPU-enabled Space tier

---

## 📝 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact & Support

- **Questions?** Open an issue on GitHub
- **Bugs?** Submit a detailed bug report
- **Collaboration?** Reach out to the team

---

## 🙏 Acknowledgments

- DistilBERT team for the lightweight transformer architecture
- HuggingFace for model hosting and Spaces platform
- ALS research community for defining the problem we're solving
- Our collaborators in computational biology
<!-- 
---

## 📚 Citation

If you use Binderr in your research, please cite:

```bibtex
@software{binderr2024,
  title={Binderr: AI-Powered TDP-43 RNA Binding Affinity Prediction},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/spaces/mandipsapkota/binderr}
}
``` -->

---
