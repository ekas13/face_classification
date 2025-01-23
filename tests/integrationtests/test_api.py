import os

from fastapi.testclient import TestClient

from face_classification.api import app

client = TestClient(app)

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Backend here!"}

def test_classify_endpoint():
    # Path to the test image
    test_image_path = "data/processed/test/person_1_face_3.jpg"

    # Ensure the test image exists
    assert os.path.exists(test_image_path), "Test image does not exist"

    # Open the test image in binary mode
    with open(test_image_path, "rb") as test_image:
        with TestClient(app) as client:
            response = client.post(
                "/classify/",
                files={"file": ("person_1_face_3.jpg", test_image, "image/jpeg")},
            )
            # Check the response status code
            assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

            # Check the response content
            response_data = response.json()
            assert "filename" in response_data, "Response does not contain 'filename'"
            assert response_data["filename"] == "person_1_face_3.jpg", "Filename does not match"
            assert "prediction" in response_data, "Response does not contain 'prediction'"
            assert "probabilities" in response_data, "Response does not contain 'probabilities'"

            # Ensure probabilities is a list
            assert isinstance(response_data["probabilities"], list), "Probabilities is not a list"