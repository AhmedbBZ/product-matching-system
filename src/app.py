

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from embedding_engine import EmbeddingEngine
from retrieval import HybridRetriever
import uvicorn

app = FastAPI(title="Product Matching API")

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


@app.get("/")
async def home():
    """Home endpoint"""
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
    # Use localhost for Windows compatibility
    uvicorn.run(app, host="127.0.0.1", port=8000)