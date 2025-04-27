FROM python:3.10

# Set working directory initially
WORKDIR /

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential wget

# Download and install TA-Lib
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr --build=`/bin/arch`-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Install Python TA-Lib and other requirements
COPY requirements.txt /
RUN pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY /app /app
COPY /config /config

# Set the entrypoint
ENTRYPOINT ["python", "app/main.py"]