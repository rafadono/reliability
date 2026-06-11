# Multi-stage build for optimized image
FROM python:3.12-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Dependencies stage
FROM base as dependencies
COPY backend/requirements.txt .
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Final stage
FROM base as final
WORKDIR /app

COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copiar el código correcto del backend
COPY backend/app.py .
COPY backend/state.py .
COPY backend/api ./api
COPY backend/models ./models
COPY backend/src ./src

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
