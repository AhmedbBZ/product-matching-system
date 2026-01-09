def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["message"] == "Product Matching API"
    assert "total_products" in response.json()


def test_search_get(client):
    response = client.get("/search?q=dog%20food&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "dog food"
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_post(client):
    request_data = {"query": "test product", "top_k": 5}
    response = client.post("/search", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test product"
    assert "results" in data
