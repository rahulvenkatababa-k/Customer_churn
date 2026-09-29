FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt && \
    pip install --no-cache-dir --upgrade "msgpack>=1.2.1"

COPY src/ ./src/
COPY deployment/ ./deployment/

EXPOSE 8000
CMD ["uvicorn", "src.networksecurity.app:app", "--host", "0.0.0.0", "--port", "8000"]
