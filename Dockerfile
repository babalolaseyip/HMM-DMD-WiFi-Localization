FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt

COPY . .

RUN mkdir -p data/raw data/processed results
RUN pip install -e .

CMD ["python", "scripts/train_model.py", "--use-synthetic", "--output-dir", "results"]
