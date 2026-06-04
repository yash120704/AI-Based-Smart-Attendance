FROM python:3.11-slim-bookworm

ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

RUN apt-get update && apt-get install -y \
    cmake build-essential libopenblas-dev \
    liblapack-dev libx11-dev libgtk-3-dev \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "cmake<4" \
    && pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir --no-deps face_recognition==1.3.0 \
    && python -c "import dlib, face_recognition; print('face_recognition import ok')"

COPY . .
CMD uvicorn api.main_api:app --host 0.0.0.0 --port ${PORT:-8000}
