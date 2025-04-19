#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p /home/sai/Desktop/StockAnalyzer/logs

# Start the news collector in the background
nohup python3 newsCollector.py > /home/sai/Desktop/StockAnalyzer/logs/collector_logs.log 2>&1 &

# Store the process ID for potential later use
echo $! > /home/sai/Desktop/StockAnalyzer/logs/collector_pid.txt

# Start following the logs
echo "News collector started with PID: $(cat /home/sai/Desktop/StockAnalyzer/logs/collector_pid.txt)"
echo "Following logs (press Ctrl+C to stop following logs while keeping the process running):"
tail -f /home/sai/Desktop/StockAnalyzer/logs/collector_logs.log
