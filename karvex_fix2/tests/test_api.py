from fastapi.testclient import TestClient

from backend.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_ask_endpoint_contract():
    response = client.post(
        "/ask",
        json={
            "query": "What is solar energy?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is solar energy?"
    assert "success" in data
    assert "answer" in data
    assert "error" in data