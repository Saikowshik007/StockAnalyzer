FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib using the .deb package
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb && \
    apt-get update && \
    apt-get install -y ./ta-lib_0.6.4_amd64.deb && \
    rm ta-lib_0.6.4_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory to the project directory
WORKDIR /home/appuser/app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install numpy && \
    pip install ta-lib && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories with relative paths and set permissions
RUN mkdir -p logs config database backups && \
    touch logs/application.log && \
    chown -R appuser:appuser /home/appuser/app && \
    chmod -R 755 /home/appuser/app && \
    chmod 777 logs/application.log

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["python", "main.py"]