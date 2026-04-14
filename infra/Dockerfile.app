FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV POETRY_EXPERIMENTAL_SYSTEM_GIT_CLIENT=true

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    "importlib-metadata<8"

RUN pip install --no-cache-dir "poetry==1.8.3"

RUN poetry config virtualenvs.create false \
    && poetry config experimental.system-git-client true \
    && poetry install --no-ansi --no-root

COPY . /app