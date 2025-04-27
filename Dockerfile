FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libffi-dev \
    libnss3 \
    libnspr4 \
    build-essential \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data persistence
RUN mkdir -p /app/logs /app/database /app/backups

# TA-Lib setup - Install from source since we're in Linux, not Windows
RUN pip install TA-Lib-Precompiled

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir TA-Lib

# Install NLTK data
RUN python -m nltk.downloader punkt

# Create a non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY . .

# Set correct ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["python", "main.py"]