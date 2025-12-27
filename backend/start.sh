#!/bin/bash
# Backend server start/restart script

PORT=8000
LOG_FILE="/tmp/backend.log"

echo "=== Back Tag Backend Server ==="

# Stop any existing server
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "Stopping existing server on port $PORT..."
    pkill -f "uvicorn.*$PORT" 2>/dev/null
    sleep 2
fi

# Change to backend directory
cd "$(dirname "$0")"

# Clear model cache for fresh start
echo "Clearing model cache..."
PYTHONPATH=. python3 -c "from app.services.fast_tagger import clear_cache; clear_cache()" 2>/dev/null

# Start server
echo "Starting server on port $PORT..."
echo "Logs: $LOG_FILE"

if [ "$1" = "--foreground" ] || [ "$1" = "-f" ]; then
    # Run in foreground (for debugging)
    PYTHONPATH=. python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
else
    # Run in background
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT > $LOG_FILE 2>&1 &
    sleep 3

    # Verify it's running
    if curl -s http://localhost:$PORT/api/health > /dev/null; then
        echo "✓ Server running at http://localhost:$PORT"
        echo "✓ Health check passed"
    else
        echo "✗ Server failed to start. Check $LOG_FILE"
        exit 1
    fi
fi
