import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Predictive Maintenance API. Use the /predict endpoint to get RUL predictions."
    }

def test_predict():
    payload = {
        "op_setting1": 2.0,
        "op_setting2": 1.0,
        "op_setting3": 3.0,
        "sensor_avg": 1500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_RUL" in json_resp
    assert isinstance(json_resp["predicted_RUL"], (int, float))
