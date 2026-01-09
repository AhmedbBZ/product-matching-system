"""
Pytest configuration and fixtures for the Product Matching System.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Provide lightweight stubs for heavy native modules that may not be
# available in the CI/test environment. This lets the test runner import
# application modules without installing large ML packages (torch, faiss, transformers, etc).
try:
    import types
    heavy_modules = ['faiss', 'torch', 'sentence_transformers', 'sklearn', 'numpy']
    for module_name in heavy_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)
except Exception:
    pass


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing without loading real models."""
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
    """
    Create a test client with mocked dependencies.
    This avoids loading real models during testing.
    """
    # Import after path is set up
    from src.app import app
    
    # Patch the global variables
    with patch('src.app.retriever', mock_retriever), \
         patch('src.app.df', mock_df), \
         patch('src.app.webbrowser'):  # Prevent browser from opening
        yield TestClient(app)


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return {
        "query": "dog food",
        "results": [
            {
                "product_id": 123,
                "title": "Rottweiler Puppy Dog Food",
                "vendor": "Royal Canine",
                "category": "Animals & Pet Supplies",
                "semantic_score": 85.5,
                "lexical_score": 80.0,
                "final_score": 83.0,
            },
            {
                "product_id": 456,
                "title": "Vet Life Growth Canine Formula",
                "vendor": "Farmina",
                "category": "Animals & Pet Supplies",
                "semantic_score": 75.0,
                "lexical_score": 70.0,
                "final_score": 73.0,
            }
        ]
    }
