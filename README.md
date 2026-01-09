# Product Matching System (MLOPS Project)

## Project Structure

```text
product-matcher/
├── .github/workflows/
│   └── tests.yml                  # CI/CD GitHub Actions
├── data/
│   ├── product_catalogue.csv       # Raw data
├── models/                         # Auto-generated
│   ├── faiss.index
│   └── metadata.pkl
├── src/
|   ├── _init_.py                  # Make src a package   
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

  ## Quick Start

  ### 1. Install Dependencies
  ```bash
  pip install -r requirements.txt
  # For testing:
  pip install -r requirements-test.txt
  ```

  ### 2. Prepare & Index Data

  We use a csv file at `data/product_catalogue.csv` with columns:
  - `product_id`, `title`, `vendor`, `tags`, `category`

  ```bash
  # 1. Clean the raw CSV
  python src/data_processing.py

  # 2. Generate embeddings
  python src/embedding_engine.py
  ```

  ### 3. Start the System

  ```bash
  python src/app.py
  ```

  The system automatically opens the browser at http://localhost:8000. This is to serve the web page.

  ## 5. Testing & CI/CD

  ### Local Testing

  ```bash
  curl "http://localhost:8000/search?q=dog%20food&top_k=5"
  ```

  ## Automated Testing

  The project includes a suite of unit tests with automated CI/CD via GitHub Actions.

  **Run tests locally:**
  ```bash
  pytest tests/ -v
  ```
  The project includes a `.github/workflows/tests.yml` file. On every push or pull request to `main` or `develop`, the system:
  1. Sets up Python 3.12.
  2. Installs testing dependencies.
  3. Runs the full test suite to ensure API stability.

  ---

  ## Evaluation

  Run the Jupyter notebook:

  ```bash
  jupyter notebook evaluation.ipynb
  ```

  This generates:
  - Recall@1, Recall@5, MRR metrics
  - Confidence calibration analysis
  - Performance visualizations

  ## Docker Deployment

  **Build for your current architecture:**
  ```bash
  docker build -t product-matcher:latest .
  ```

  **Run the container:**
  ```bash
  docker run -p 8000:8000 product-matcher:latest
  ```

  ---

## API Documentation

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

## Configuration

Edit `src/retrieval.py` to adjust:
- `semantic_weight` (default: 0.7)
- `lexical_weight` (default: 0.3)
- `threshold_high` (default: 85)

## Performance

- **Recall@1**: 96.6%
- **Recall@5**: 94.8%
- **MMR**: 95.7%
- **Dataset**: 5,270 products

## Tech Stack
- **Backend:** Python 3.12, FastAPI, Uvicorn
- **AI/Search:** Sentence-Transformers (All-MiniLM-L6-v2), FAISS, BM25
- **Frontend:** HTML5/JavaScript
- **DevOps:** Docker, GitHub Actions, Pytest

## Authors

[Oussama Mrabtini / Mouhamed Ghassan Ayyari / Ahmed Bouzid]  
MLOPS-Project-2

---

### Implementation Notes:
1.  **Frontend:** The `app.py` is configured to serve `index.html` from the root directory via `FileResponse`.
2.  **Mocks:** In `conftest.py`, we mock `torch` and `sentence_transformers`. This allows us to run tests in environments like GitHub Actions.