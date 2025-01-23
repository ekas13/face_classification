FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/face_classification/frontend.py /app/frontend.py
ENV BACKEND=https://app-docker-image-294894715547.europe-west1.run.app

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "8080", "--server.address=0.0.0.0"]
