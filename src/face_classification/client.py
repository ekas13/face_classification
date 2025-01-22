import argparse

import requests


class FaceClassificationClient:
    """Client for communicating with the Face Classification API."""

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


def main():
    parser = argparse.ArgumentParser(description="Face Classification Client")
    parser.add_argument(
        "--base_url", type=str, default="http://localhost:8000", help="Base URL for the Face Classification API"
    )
    args = parser.parse_args()

    base_url = args.base_url
    client = FaceClassificationClient(base_url=base_url)

    while True:
        command = input("Enter command (train, evaluate, predict, exit): ").strip().lower()
        if command == "train":
            print(client.train_model())
        elif command == "evaluate":
            print(client.evaluate_model())
        elif command == "predict":
            image_path = input("Enter image path: ").strip()
            print(client.predict_single_image(image_path))
        elif command == "exit":
            break
        else:
            print("Unknown command. Please enter 'train', 'evaluate', 'predict', or 'exit'.")


if __name__ == "__main__":
    main()
