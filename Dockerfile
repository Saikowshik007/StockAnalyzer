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

# Install TA-Lib from source with explicit library path
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Ensure the libraries are in the correct place and update cache
RUN ln -s /usr/local/lib/libta_lib.so.0 /usr/lib/ && \
    ln -s /usr/local/lib/libta_lib.so.0.0.0 /usr/lib/ && \
    ln -s /usr/local/lib/libta_lib.so /usr/lib/ && \
    ldconfig

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (install TA-Lib separately)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir -r requirements.txt

# Install TA-Lib Python wrapper
RUN pip install --global-option=build_ext --global-option="-L/usr/local/lib/" --global-option="-I/usr/local/include" ta-lib

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