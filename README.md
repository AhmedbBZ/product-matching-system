To integrate your new `app.py`, `Dockerfile`, and testing suite into the project documentation, I have updated the `README.md` to reflect the new structure, the automated web interface, and the CI/CD pipeline.

Replace your current **README.md** with this version:

---

# Product Matching System (MLOPS Project)

A semantic search system that matches product queries against a shop catalogue using AI embeddings and keyword search. This project includes a FastAPI backend, a web interface, and a full CI/CD testing pipeline.

## 📁 Project Structure

```text
product-matcher/
├── .github/workflows/
│   └── tests.yml                  # CI/CD GitHub Actions
├── data/
│   ├── product_catalogue.csv       # Raw data
│   └── product_catalogue_processed.csv # Generated after cleaning
├── models/                         # Auto-generated
│   ├── faiss.index
│   └── metadata.pkl
├── src/
│   ├── data_processing.py         # Data cleaning
│   ├── embedding_engine.py        # Embeddings + FAISS
│   ├── retrieval.py               # Hybrid search
│   └── app.py                     # FastAPI REST API & UI Router
├── tests/                          # Automated Testing
│   ├── conftest.py                # Pytest configuration/mocks
│   └── test_app.py                # API unit tests
├── index.html                      # Web Interface (Frontend)
├── requirements.txt               # Main dependencies
├── requirements-test.txt          # Testing dependencies
├── Dockerfile                     # Containerization setup
├── README.md                      # This file
└── REPORT.md                      # Technical report
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# For development/testing:
pip install -r requirements-test.txt
```

### 2. Prepare & Index Data
You must process the data and build the FAISS index before running the API.
```bash
# 1. Clean the raw CSV
python src/data_processing.py

# 2. Generate embeddings (requires models/ folder)
python src/embedding_engine.py
```

### 3. Start the System
```bash
python src/app.py
```
*   **API:** http://localhost:8000/api/status
*   **Web UI:** The system will automatically open your browser to http://localhost:8000

---

## 🧪 Testing & CI/CD

### Local Testing
We use `pytest` with mocks for heavy dependencies (like Torch and FAISS) to ensure fast testing.
```bash
pytest tests/ -v
```

### GitHub Actions
The project includes a `.github/workflows/tests.yml` file. On every **push** or **pull request** to `main` or `develop`, the system:
1. Sets up Python 3.12.
2. Installs testing dependencies.
3. Runs the full test suite to ensure API stability.

---

## 🐳 Docker Deployment

The system is containerized for easy deployment. It includes a health check and runs as a non-root user for security.

**Build for your current architecture:**
```bash
docker build -t product-matcher .
```

**Run the container:**
```bash
docker run -p 8000:8000 product-matcher
```

---

## 📖 API Documentation

### Search (Hybrid)
**Endpoint:** `GET /search?q=query&top_k=5`  
**Endpoint:** `POST /search` (JSON Body: `{"query": "...", "top_k": 5}`)

**Response Example:**
```json
{
  "query": "dog food",
  "results": [
    {
      "product_id": 4428755271778,
      "title": "Rottewiler Puppy Dog Food",
      "semantic_score": 70.63,
      "lexical_score": 59.96,
      "confidence": 67.43,
      "match_quality": "MEDIUM"
    }
  ]
}
```

### Health Check
**Endpoint:** `GET /api/status`  
Returns the total number of products loaded in memory and system status.

---

## 🛠️ Tech Stack
- **Backend:** Python 3.12, FastAPI, Uvicorn
- **AI/Search:** Sentence-Transformers (All-MiniLM-L6-v2), FAISS, BM25
- **Frontend:** HTML5/JavaScript (Tailwind/Fetch API)
- **DevOps:** Docker, GitHub Actions, Pytest

## 📝 Authors
**Oussama Mrabtini / Mouhamed Ghassan Ayyari / Ahmed Bouzid**  
*FSB MLOPS-Project 2026*

---

### 💡 Implementation Notes for the Team:
1.  **Frontend:** The `app.py` is configured to serve `index.html` from the root directory via `FileResponse`.
2.  **Mocks:** In `conftest.py`, we mock `torch` and `sentence_transformers`. This allows us to run tests in environments without a GPU or high RAM (like GitHub Actions).
3.  **Docker Platform:** If deploying to a server (AMD64) from a Mac (ARM64), remember to use `docker build --platform linux/amd64 -t product-matcher .`.