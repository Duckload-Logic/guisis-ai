import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.getcwd())

from src.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_prediction():
    with TestClient(app) as client:
        payload = {
            "text": "Natatakot po ako pumasok sa PUP kasi feeling ko "
            + "hindi ko kaya lahat ng requirements."
        }
        response = client.post("/api/v1/classify", json=payload)

        print("\nPrediction Result:")
        print(response.json())

        assert response.status_code == 200
        assert "label" in response.json()
        assert "confidence" in response.json()
        assert response.json()["label"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

if __name__ == "__main__":
    # Self-run if called directly
    try:
        print("Running Health Check...")
        test_health()
        print("Health Check Passed!")

        print("\nRunning Prediction Test...")
        test_prediction()
        print("Prediction Test Passed!")
    except Exception as e:
        print(f"\nTests Failed: {e}")
        sys.exit(1)
