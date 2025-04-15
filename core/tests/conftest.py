"""
Pytest configuration for the financial analysis dashboard.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app
from main import app

@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app.
    """
    return TestClient(app)
