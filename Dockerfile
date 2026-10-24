FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESUMEPROOF_USE_FAKE_LLM=true

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "pytest", "tests/", "-v"]
