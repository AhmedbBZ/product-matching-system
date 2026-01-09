import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


class EmbeddingEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # all-MiniLM-L6-v2 is a balanced and performing model

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
        
        self.index = None
        self.id_mapping = {}
        self.metadata = {}
        
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_index(self, df: pd.DataFrame, text_column: str = 'searchable_text'):
        print("\n=== Building FAISS Index ===")
        
        texts = df[text_column].tolist()
        
        embeddings = self.create_embeddings(texts)
        
        faiss.normalize_L2(embeddings)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        self.id_mapping = {i: row['product_id'] for i, row in df.iterrows()}
        self.metadata = df.to_dict('index')
        
        print(f"Index built with {self.index.ntotal} vectors")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:

        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
                
            product_id = self.id_mapping[idx]
            metadata = self.metadata[idx]
            
            results.append({
                'product_id': product_id,
                'title': metadata['title'],
                'vendor': metadata['vendor'],
                'category': metadata['category'],
                'searchable_text': metadata['searchable_text'],
                'similarity_score': float(score),
                'confidence': self._score_to_confidence(float(score))
            })
        
        return results
    
    def _score_to_confidence(self, score: float) -> float:
        confidence = score * 100
        return round(max(0, min(100, confidence)), 2)
    
    def save_index(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        metadata_path = os.path.join(index_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_mapping': self.id_mapping,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
    
    def load_index(self, index_dir: str):
        index_path = os.path.join(index_dir, 'faiss.index')
        self.index = faiss.read_index(index_path)
        
        # Load mappings and metadata
        metadata_path = os.path.join(index_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.id_mapping = data['id_mapping']
            self.metadata = data['metadata']
            self.dimension = data['dimension']


if __name__ == "__main__":
    df = pd.read_csv('data/product_catalogue_processed.csv')
    
    engine = EmbeddingEngine()
    engine.build_index(df)
    engine.save_index('models')
    
    print("\n✓ Index built successfully!")