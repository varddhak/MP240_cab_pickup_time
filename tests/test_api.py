from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["target"] == "pickup_eta_min"

def test_prediction():
    payload = {
        "drivers_nearby": 5,
        "distance_to_driver_km": 2.5,
        "hour": 14,
        "rain_mm": 2,
        "surge": 1,
        "road_closures": 0,
        "car_type": "Sedan",
        "zone": "Old City"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction_minutes" in response.json()
    assert response.json()["prediction_minutes"] >= 0
