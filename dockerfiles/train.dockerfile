# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY configs/ configs/
COPY models/ models/
COPY pyproject.toml pyproject.toml
COPY README.md README.md

WORKDIR /
#maybe we'll need to add a directory for training outputs if we're gonna save them
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -e . --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/face_classification/train.py"]
