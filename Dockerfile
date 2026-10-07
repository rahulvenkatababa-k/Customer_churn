FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

EXPOSE 8000

CMD [ "uvicorn" , "src.customer_churn.app:app" , "--host" , "0.0.0.0" , "--port", "8000" ]

