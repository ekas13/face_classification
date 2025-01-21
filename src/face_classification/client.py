import requests

class FaceClassificationClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def predict_single_image(self, image_path):
        url = f"{self.base_url}/predict_single_image"
        params = {"image_path": image_path}
        response = requests.get(url, params=params)
        return response.json()

    def train_model(self):
        url = f"{self.base_url}/train_model"
        response = requests.get(url)
        return response.json()

    def evaluate_model(self):
        url = f"{self.base_url}/evaluate_model"
        response = requests.get(url)
        return response.json()

# Example usage:
if __name__ == "__main__":
    client = FaceClassificationClient()
    # print(client.predict_single_image("data/processed/test/person_1_face_3.jpg"))
    print(client.train_model())
    print(client.evaluate_model())