import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    # Uygulamanın başlangıç işlemlerini çalıştırır.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client):
    message = {
        "text": "The application keeps crashing when I log in."
    }

    response = client.post("/predict", json=message)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in {
        "Incident", "Problem", "Request", "Change"
    }

    assert set(result["scores"]) == {
        "Incident", "Problem", "Request", "Change"
    }

    assert all(
        0 <= score <= 1
        for score in result["scores"].values()
    )


def test_empty_message(client):
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_whitespace_message(client):
    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    assert response.status_code == 422