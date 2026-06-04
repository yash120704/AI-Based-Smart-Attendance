FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    cmake build-essential libopenblas-dev \
    liblapack-dev libx11-dev libgtk-3-dev \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .
CMD uvicorn api.main_api:app --host 0.0.0.0 --port ${PORT:-8000}
