import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "face_classification"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements_tests.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements_frontend.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


# call with invoke evaluate --model-path = <path here>
@task
def evaluate(ctx: Context, model_path: str = None) -> None:
    """Run evaluation with optional model path."""
    if model_path is None:
        ctx.run(f"python src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)
    else:
        ctx.run(f"python src/{PROJECT_NAME}/evaluate.py {model_path}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def server(ctx: Context) -> None:
    """Run the API server."""
    ctx.run("uvicorn src.face_classification.api:app --host 0.0.0.0 --port 8000 --reload", echo=True, pty=not WINDOWS)


@task
def frontend(ctx: Context) -> None:
    """Run the frontend server."""
    ctx.run("streamlit run src/face_classification/frontend.py --server.port 8080 --server.address=0.0.0.0", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )