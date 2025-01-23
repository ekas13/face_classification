import os

import pandas as pd
import requests
import streamlit as st


def get_backend_url():
    """Get the URL of the backend service."""
    return os.environ.get("BACKEND", "http://localhost:8000")


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image}, timeout=30)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"][0][0]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write(f"Prediction: this is face {prediction}.")

            # make a nice bar chart
            data = {"Class": [i for i in range(len(probabilities))], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
