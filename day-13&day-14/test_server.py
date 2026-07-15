import pytest
from fastapi.testclient import TestClient

from day14_server import app

client = TestClient(app)


def test_successful_http_rag_assembly():
    payload = {
        "query": "How do we deploy to Tokyo-West region?",
        "context_chunks": ["Tokyo-West utilizes cluster groups alpha and beta."],
    }

    response = client.post("/api/v1/assemble", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "Tokyo-West utilizes cluster groups" in data["compiled_prompt"]


def test_server_rejects_whitespace_query():
    payload = {
        "query": "   ",
        "context_chunks": ["Some text"],
    }

    response = client.post("/api/v1/assemble", json=payload)

    assert response.status_code == 422


def test_server_accepts_empty_context():
    payload = {
        "query": "What is RAG?",
        "context_chunks": [],
    }

    response = client.post(
        "/api/v1/assemble",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "[No Context Included]" in data["compiled_prompt"]
    assert "What is RAG?" in data["compiled_prompt"]


def test_server_rejects_too_many_context_chunks():
    payload = {
        "query": "valid query",
        "context_chunks": ["chunk"] * 21,
    }

    response = client.post("/api/v1/assemble", json=payload)

    assert response.status_code == 422


def test_server_rejects_query_over_limit():
    payload = {
        "query": "a" * 501,
        "context_chunks": ["chunk"],
    }

    response = client.post("/api/v1/assemble", json=payload)

    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
