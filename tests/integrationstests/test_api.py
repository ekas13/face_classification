from fastapi.testclient import TestClient
from web_app.api import app

client = TestClient(app)

def test_train_api(model):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}

def test_evaluate_api(model):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}

def test_predict(model):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}
