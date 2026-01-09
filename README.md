# Product Matching System

A semantic search system that matches product queries against a shop catalogue using AI embeddings and keyword search.

## 📁 Project Structure

```
product-matcher/
├── data/
│   └── product_catalogue.csv       # Your product data
├── models/                         # Auto-generated
│   ├── faiss.index
│   └── metadata.pkl
├── src/
│   ├── data_processing.py         # Data cleaning
│   ├── embedding_engine.py        # Embeddings + FAISS
│   ├── retrieval.py               # Hybrid search
│   └── app.py                     # REST API
├── evaluation.ipynb               # Metrics & evaluation
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── REPORT.md                      # Technical report
└── Dockerfile                     # Docker setup
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV file at `data/product_catalogue.csv` with columns:
- `product_id`, `title`, `vendor`, `tags`, `category`

### 3. Process Data & Build Index

```bash
# Clean the data
python src/data_processing.py

# Build embeddings
python src/embedding_engine.py
```

### 4. Start API

```bash
python src/app.py
```

API runs at http://localhost:8000

### 5. Test It

```bash
curl "http://localhost:8000/search?q=dog%20food&top_k=5"
```

## 📊 Evaluation

Run the Jupyter notebook:

```bash
jupyter notebook evaluation.ipynb
```

This generates:
- Recall@1, Recall@5, MRR metrics
- Confidence calibration analysis
- Performance visualizations

## 🐳 Docker

```bash
docker build -t product-matcher .
docker run -p 8000:8000 product-matcher
```

## 📖 API Endpoints

### Search Products
```bash
GET /search?q=YOUR_QUERY&top_k=2
POST /search
```

**Response:**
```json
{
  "query": "dog food",
  "results": [
    {
      "product_id": 4428755271778,
      "title": "Rottewiler Puppy Dog Food",
      "vendor": "royal canine",
      "category": "Animals & Pet Supplies",
      "searchable_text": "Rottewiler Puppy Dog Food royal canine Animals & Pet Supplies dog food dry dog food",
      "semantic_score": 70.63,
      "lexical_score": 59.96,
      "final_score": 67.43,
      "confidence": 67.43,
      "match_quality": "MEDIUM",
      "explanation": "Strong semantic similarity + Partial keyword match"
    },
    {
      "product_id": 4373493645410,
      "title": "Vet Life Growth Canine Formula Dog Food",
      "vendor": "Farmina",
      "category": "Animals & Pet Supplies",
      "searchable_text": "Vet Life Growth Canine Formula Dog Food Farmina Animals & Pet Supplies 12 kg 2 kg dog food dog pregnancy care dog sexual care",
      "semantic_score": 64.19,
      "lexical_score": 54.55,
      "final_score": 61.3,
      "confidence": 61.3,
      "match_quality": "MEDIUM",
      "explanation": "Moderate semantic similarity + Partial keyword match"
    }
  ]
}
```

## 🔧 Configuration

Edit `src/retrieval.py` to adjust:
- `semantic_weight` (default: 0.7)
- `lexical_weight` (default: 0.3)
- `threshold_high` (default: 85)

## 📈 Performance

- **Recall@1**: 96.6%
- **Recall@5**: 94.8%
- **MMR**: 95.7%
- **Dataset**: 5,270 products

## 🛠️ Tech Stack

- Python 3.12
- FastAPI
- sentence-transformers
- FAISS
- BM25

## 📝 Author

[Oussama Mrabtini/Mouhamed Ghassan Ayyari/Ahmed Bouzid] FSB MLOPS-Project 2026