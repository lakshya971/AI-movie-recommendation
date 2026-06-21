#!/bin/bash

echo "Starting FastAPI backend..."
python -m uvicorn main:app --host 127.0.0.1 --port 8000 &

# Wait for backend to start up (polling health check)
echo "Waiting for backend to initialize..."
for i in {1..15}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "Backend is ready!"
    break
  fi
  echo "Backend not ready yet, sleeping 2s (attempt $i/15)..."
  sleep 2
done

# Check if backend is up
if ! curl -s http://127.0.0.1:8000/health > /dev/null; then
  echo "WARNING: Backend failed to start on time."
fi

echo "Starting Streamlit frontend on port $PORT..."
python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

