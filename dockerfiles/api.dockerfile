# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY configs/ configs/
COPY models/ models/
COPY pyproject.toml pyproject.toml
COPY README.md README.md


RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -e . --no-deps --no-cache-dir --verbose

CMD ["uvicorn", "src.face_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
