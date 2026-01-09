# Setup Guide (Student Version)


## 🚀 Step 1: Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Process data
python src/data_processing.py

# Build embeddings 
python src/embedding_engine.py

# Start API
python src/app.py
```

## ✅ Step 2: Test

```bash
# In browser or curl
curl "http://localhost:8000/search?q=dog%20food&top_k=5"
```

## 📊 Step 5: Evaluation

```bash
jupyter notebook evaluation.ipynb
# Run all cells
```

## 🐳 Docker (Optional)

```bash
docker build -t product-matcher .
docker run -p 8000:8000 product-matcher
```

---

## Final Structure

```
product-matcher/
├── data/
│   ├── product_catalogue.csv          
│   └── product_catalogue_processed.csv # Generated
├── models/                            # Generated
│   ├── faiss.index
│   └── metadata.pkl
├── src/
│   ├── data_processing.py
│   ├── embedding_engine.py
│   ├── retrieval.py
│   └── app.py
├── evaluation.ipynb
├── requirements.txt
├── README.md
├── REPORT.md
└── Dockerfile
```

