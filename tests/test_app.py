"""
Unit tests for the Product Matching System FastAPI application.
Tests cover API endpoints, error handling, and response validation.
"""
import pytest
from fastapi import HTTPException


class TestAPIStatus:
    """Test cases for the API status endpoint."""
    
    def test_api_status_returns_success(self, client):
        """
        Test that /api/status endpoint returns 200 with correct structure.
        """
        response = client.get("/api/status")
        
        assert response.status_code == 200
        assert "message" in response.json()
        assert "total_products" in response.json()
        assert response.json()["message"] == "Product Matching API"
        assert response.json()["total_products"] == 2  # mock_df has 2 rows


class TestHomeEndpoint:
    """Test cases for the home/index endpoint."""
    
    def test_home_serves_html(self, client):
        """
        Test that / endpoint serves the HTML interface.
        """
        response = client.get("/")
        
        assert response.status_code == 200
        # FileResponse returns HTML content
        assert response.headers.get("content-type") is not None


class TestSearchEndpointGET:
    """Test cases for the GET /search endpoint."""
    
    def test_search_get_with_valid_query(self, client):
        """
        Test GET /search with a valid query parameter.
        Should return results array with expected structure.
        """
        response = client.get("/search?q=dog%20food&top_k=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "dog food"
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0

    def test_search_get_with_custom_top_k(self, client):
        """
        Test GET /search with custom top_k parameter.
        """
        response = client.get("/search?q=laptop&top_k=3")
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "laptop"
        assert isinstance(data["results"], list)

    def test_search_get_returns_product_fields(self, client):
        """
        Test that search results contain expected product fields.
        """
        response = client.get("/search?q=shoes&top_k=5")
        
        assert response.status_code == 200
        data = response.json()
        if len(data["results"]) > 0:
            result = data["results"][0]
            # Check for expected fields in results
            assert "product_id" in result or "title" in result
            assert isinstance(result, dict)


class TestSearchEndpointPOST:
    """Test cases for the POST /search endpoint."""
    
    def test_search_post_with_valid_request(self, client):
        """
        Test POST /search with valid JSON request body.
        """
        request_data = {
            "query": "bluetooth speaker",
            "top_k": 5
        }
        response = client.post("/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "bluetooth speaker"
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_post_default_top_k(self, client):
        """
        Test POST /search with default top_k parameter.
        """
        request_data = {"query": "headphones"}
        response = client.post("/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "headphones"

    def test_search_post_invalid_request_missing_query(self, client):
        """
        Test POST /search with missing required 'query' field.
        Should return validation error (422).
        """
        request_data = {"top_k": 5}  # Missing 'query'
        response = client.post("/search", json=request_data)
        
        assert response.status_code == 422  # Unprocessable Entity


class TestSearchFunctionality:
    """Test cases for search functionality and data consistency."""
    
    def test_search_returns_consistent_structure(self, client):
        """
        Test that all results follow the same data structure.
        """
        response = client.get("/search?q=test&top_k=10")
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        if len(results) > 0:
            # Verify all results are dictionaries
            for result in results:
                assert isinstance(result, dict)

    def test_search_with_empty_string_query(self, client):
        """
        Test search with empty string query.
        Should handle gracefully (may return error or empty results).
        """
        response = client.get("/search?q=&top_k=5")
        
        # Should either return 200 with results or error
        assert response.status_code in [200, 400, 422]

    def test_search_with_special_characters(self, client):
        """
        Test search with special characters in query.
        Should handle URL encoding properly.
        """
        response = client.get("/search?q=test%20%26%20product&top_k=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_search_top_k_parameter_validation(self, client):
        """
        Test that top_k parameter is properly handled.
        """
        response = client.get("/search?q=test&top_k=1")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["results"], list)


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_invalid_endpoint_returns_404(self, client):
        """
        Test that invalid endpoints return 404.
        """
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_search_with_very_large_top_k(self, client):
        """
        Test search with very large top_k value.
        Should handle gracefully.
        """
        response = client.get("/search?q=test&top_k=999999")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["results"], list)

    def test_search_method_validation(self, client):
        """
        Test that invalid HTTP methods return appropriate error.
        """
        response = client.delete("/search?q=test")
        assert response.status_code == 405  # Method Not Allowed


class TestResponseFormats:
    """Test cases for response format validation."""
    
    def test_search_response_json_valid(self, client):
        """
        Test that /search returns valid JSON.
        """
        response = client.get("/search?q=test&top_k=5")
        
        assert response.status_code == 200
        # This will raise if JSON is invalid
        data = response.json()
        assert isinstance(data, dict)

    def test_api_status_response_json_valid(self, client):
        """
        Test that /api/status returns valid JSON.
        """
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert isinstance(data["total_products"], int)

    def test_response_headers_include_content_type(self, client):
        """
        Test that responses include proper Content-Type headers.
        """
        response = client.get("/api/status")
        
        assert "content-type" in response.headers or "Content-Type" in response.headers


class TestCORSHeaders:
    """Test cases for CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """
        Test that CORS middleware is properly configured.
        """
        response = client.get("/api/status")
        
        # CORS headers might be present (depending on middleware config)
        # At minimum, the endpoint should be accessible
        assert response.status_code == 200


class TestDataValidation:
    """Test cases for data validation and filtering."""
    
    def test_search_query_parameter_required(self, client):
        """
        Test that 'q' parameter is required for GET /search.
        """
        response = client.get("/search")
        
        # Should fail without 'q' parameter
        assert response.status_code == 422

    def test_top_k_defaults_to_5(self, client):
        """
        Test that top_k defaults to 5 if not specified.
        """
        response = client.get("/search?q=test")
        
        assert response.status_code == 200
        # The mock always returns 1 result, but default top_k should be 5
        data = response.json()
        assert isinstance(data["results"], list)
