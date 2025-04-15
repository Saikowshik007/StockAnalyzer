#!/bin/bash

# Start the news collector in the background
nohup python3 newsCollector.py > collector_logs.log 2>&1 &

# Store the process ID for potential later use
echo $! > collector_pid.txt

# Start following the logs
echo "News collector started with PID: $(cat collector_pid.txt)"
echo "Following logs (press Ctrl+C to stop following logs while keeping the process running):"
tail -f collector_logs.log