import sys
from pathlib import Path

# Add project root to path so imports work from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.embedding_engine import EmbeddingEngine
from src.retrieval import HybridRetriever
import uvicorn
import webbrowser
import os
import threading
import time

app = FastAPI(title="Product Matching API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
retriever = None
df = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: List[dict]


@app.on_event("startup")
async def load_models():
    """Load data and models on startup"""
    global retriever, df
    
    print("Loading data and models...")
    
    # Load processed data
    df = pd.read_csv('data/product_catalogue_processed.csv')
    print(f"✓ Loaded {len(df)} products")
    
    # Load embedding engine
    engine = EmbeddingEngine()
    engine.load_index('models')
    print("✓ Loaded embeddings")
    
    # Create retriever
    retriever = HybridRetriever(engine, df)
    print("✓ System ready!")
    
    # Open web browser automatically
    try:
        def open_browser():
            time.sleep(2)  # Wait for server to be ready
            webbrowser.open('http://127.0.0.1:8000')
            print("✓ Opened browser at http://127.0.0.1:8000")
        
        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()
    except Exception as e:
        print(f"Warning: Could not open browser automatically: {e}")


@app.get("/")
async def home():
    """Serve the web interface"""
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index.html'))
    return FileResponse(index_path)


@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "message": "Product Matching API",
        "total_products": len(df) if df is not None else 0
    }


@app.get("/search")
async def search_get(q: str, top_k: int = 5):
    """Search products - GET method"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    results = retriever.search_hybrid(q, k=top_k)
    
    return {
        "query": q,
        "results": results
    }


@app.post("/search")
async def search_post(request: SearchRequest):
    """Search products - POST method"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    results = retriever.search_hybrid(request.query, k=request.top_k)
    
    return {
        "query": request.query,
        "results": results
    }


if __name__ == "__main__":
    # Allow host/port to be configured via environment (useful for Docker)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)