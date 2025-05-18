FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY mlflow_artifacts/ mlflow_artifacts/

EXPOSE 8000

ENV IN_DOCKER=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]