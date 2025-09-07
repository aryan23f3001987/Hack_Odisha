# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.11-slim AS builder

# Prevent warnings
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set workdir
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Stage 2: Final image
# -------------------------
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy only installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy app code
COPY . .

# Expose port for Railway
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
