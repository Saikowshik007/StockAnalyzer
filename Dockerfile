FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create directories for data persistence
RUN mkdir -p /app/logs /app/database /app/backups

RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz
RUN tar -xvf ta-lib-0.4.0-src.tar.gz
WORKDIR /ta-lib
RUN ./configure --prefix=/usr --build=`/bin/arch`-unknown-linux-gnu
RUN make
RUN make install
RUN pip install --no-cache-dir TA-Lib
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