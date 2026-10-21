FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port (Hugging Face Spaces uses 7860, Cloud platforms use $PORT)
ENV PORT=7860
EXPOSE 7860

CMD ["python", "web_app.py"]
