FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data persistence
RUN mkdir -p /app/logs /app/database /app/backups

# Install TA-Lib using the .deb package
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb && \
    apt-get update && \
    apt-get install -y ./ta-lib_0.6.4_amd64.deb && \
    rm ta-lib_0.6.4_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ta-lib

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