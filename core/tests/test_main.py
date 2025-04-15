"""
Tests for the main FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

def test_root(client):
    """Test that the root endpoint returns the index.html file."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_static_files(client):
    """Test that static files are served correctly."""
    response = client.get("/static/css/styles.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]

    response = client.get("/static/js/dashboard.js")
    assert response.status_code == 200
    assert "application/javascript" in response.headers["content-type"]
