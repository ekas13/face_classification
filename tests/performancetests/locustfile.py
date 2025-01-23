import os
from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A Locust user class simulating interactions with the FastAPI app."""

    wait_time = between(1, 2)

    @task(1)
    def get_root(self) -> None:
        """Simulate a user visiting the root URL of the FastAPI app."""
        response = self.client.get("/")
        if response.status_code != 200:
            print(f"Failed to access root endpoint. Status code: {response.status_code}")
        else:
            print("Root endpoint accessed successfully.")

    @task(3)
    def classify_image(self) -> None:
        """Simulate a user uploading an image to the classify endpoint."""
        # Path to the test image
        test_image_path = "data/processed/test/person_1_face_3.jpg"

        # Ensure the test image exists
        if not os.path.exists(test_image_path):
            print("Test image does not exist. Skipping classify test.")
            return

        # Open the test image in binary mode
        with open(test_image_path, "rb") as test_image:
            files = {"file": ("person_1_face_3.jpg", test_image, "image/jpeg")}
            response = self.client.post("/classify/", files=files)
            if response.status_code != 200:
                print(f"Failed to classify image. Status code: {response.status_code}")
            else:
                response_data = response.json()
                print("Classify response:", response_data)
