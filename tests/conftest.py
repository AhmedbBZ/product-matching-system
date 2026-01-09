"""
Pytest configuration and fixtures for the Product Matching System.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock heavy dependencies before importing app
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    retriever = Mock()
    retriever.search_hybrid = Mock(return_value=[
        {
            "product_id": 123,
            "title": "Test Dog Food",
            "vendor": "Test Vendor",
            "category": "Animals & Pet Supplies",
            "semantic_score": 85.5,
            "lexical_score": 80.0,
            "final_score": 83.0,
            "confidence": 83.0,
            "match_quality": "HIGH",
            "explanation": "Strong match"
        }
    ])
    return retriever


@pytest.fixture
def mock_df():
    """Mock DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        "product_id": [123, 456],
        "title": ["Test Product 1", "Test Product 2"],
        "vendor": ["Vendor A", "Vendor B"]
    })


@pytest.fixture
def client(mock_retriever, mock_df):
    """Create a test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    from src.app import app
    
    # Patch the global variables and browser opening
    with patch('src.app.retriever', mock_retriever), \
         patch('src.app.df', mock_df), \
         patch('src.app.webbrowser'):
        yield TestClient(app)

