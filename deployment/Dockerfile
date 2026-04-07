FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY backend /app/backend
COPY frontend /app/frontend
COPY scraper /app/scraper
COPY scripts /app/scripts
COPY data/mock /app/data/mock

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8000 8501

CMD ["bash", "-lc", "uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
