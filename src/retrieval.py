import pandas as pd
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from .embedding_engine import EmbeddingEngine
import re


class HybridRetriever:
    # Combine semantic and lexical search
    
    def __init__(self, 
                 embedding_engine: EmbeddingEngine,
                 df: pd.DataFrame,
                 semantic_weight: float = 0.7,
                 lexical_weight: float = 0.3):

        self.embedding_engine = embedding_engine
        self.df = df
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        
        print("Building BM25 index...")
        self._build_bm25_index()
        
        self.threshold_high = 85
        self.threshold_medium = 60
        self.threshold_low = 40
        
    def _build_bm25_index(self):
        tokenized_docs = []
        for text in self.df['searchable_text']:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        self.tokenized_docs = tokenized_docs
        print(f"BM25 index built with {len(tokenized_docs)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search_semantic(self, query: str, k: int = 10) -> List[Dict]:
        return self.embedding_engine.search(query, k=k)
    
    def search_lexical(self, query: str, k: int = 10) -> List[Dict]:
        query_tokens = self._tokenize(query)
        
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            
            row = self.df.iloc[idx]
            results.append({
                'product_id': row['product_id'],
                'title': row['title'],
                'vendor': row['vendor'],
                'category': row['category'],
                'searchable_text': row['searchable_text'],
                'bm25_score': float(score),
                'confidence': self._normalize_bm25_score(float(score))
            })
        
        return results
    
    def _normalize_bm25_score(self, score: float) -> float:
        normalized = 100 * (1 - np.exp(-score / 10))
        return round(max(0, min(100, normalized)), 2)
    
    def search_hybrid(self, query: str, k: int = 5) -> List[Dict]:
        k_retrieve = min(k * 3, 20)
        
        semantic_results = self.search_semantic(query, k=k_retrieve)
        lexical_results = self.search_lexical(query, k=k_retrieve)
        
        combined_scores = {}
        
        for result in semantic_results:
            pid = result['product_id']
            semantic_score = result['similarity_score'] * 100
            combined_scores[pid] = {
                'semantic_score': semantic_score,
                'lexical_score': 0,
                'metadata': result
            }
        
        for result in lexical_results:
            pid = result['product_id']
            lexical_score = result['confidence']
            
            if pid in combined_scores:
                combined_scores[pid]['lexical_score'] = lexical_score
            else:
                combined_scores[pid] = {
                    'semantic_score': 0,
                    'lexical_score': lexical_score,
                    'metadata': result
                }
        
        final_results = []
        for pid, scores in combined_scores.items():
            final_score = (
                self.semantic_weight * scores['semantic_score'] +
                self.lexical_weight * scores['lexical_score']
            )
            
            metadata = scores['metadata']
            
            final_results.append({
                'product_id': pid,
                'title': metadata['title'],
                'vendor': metadata['vendor'],
                'category': metadata['category'],
                'searchable_text': metadata['searchable_text'],
                'semantic_score': round(scores['semantic_score'], 2),
                'lexical_score': round(scores['lexical_score'], 2),
                'final_score': round(final_score, 2),
                'confidence': round(final_score, 2),
                'match_quality': self._get_match_quality(final_score),
                'explanation': self._generate_explanation(scores, query)
            })
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:k]
    
    def _get_match_quality(self, score: float) -> str:
        if score >= self.threshold_high:
            return "HIGH"
        elif score >= self.threshold_medium:
            return "MEDIUM"
        elif score >= self.threshold_low:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_explanation(self, scores: Dict, query: str) -> str:
        semantic = scores['semantic_score']
        lexical = scores['lexical_score']
        
        explanations = []
        
        if semantic > 70:
            explanations.append("Strong semantic similarity")
        elif semantic > 40:
            explanations.append("Moderate semantic similarity")
        
        if lexical > 70:
            explanations.append("Strong keyword match")
        elif lexical > 40:
            explanations.append("Partial keyword match")
        
        if not explanations:
            explanations.append("Weak match")
        
        return " + ".join(explanations)


if __name__ == "__main__":
    print("Loading data and models...")
    df = pd.read_csv('data/processed/product_catalogue_processed.csv')
    
    engine = EmbeddingEngine()
    engine.load_index('models/embeddings')
    
    retriever = HybridRetriever(engine, df)
    
    test_queries = [
        "dog food for large breeds",
        "toy for pets",
        "leather leash",
        "chicken flavored treats"
    ]
    
    print("\n=== Testing Hybrid Search ===")
    for query in test_queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 80)
        results = retriever.search_hybrid(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Vendor: {result['vendor']}")
            print(f"   Final Score: {result['final_score']}% ({result['match_quality']})")
            print(f"   Semantic: {result['semantic_score']}% | Lexical: {result['lexical_score']}%")
            print(f"   Explanation: {result['explanation']}")