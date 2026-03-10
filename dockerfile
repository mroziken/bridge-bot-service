FROM python:3.11-slim@sha256:7c6c39da0b3dfb83d53a4c68e83f3c6b0e5e5f5e5e5e5e5e5e5e5e5e5e5e5e5

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]