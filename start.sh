#!/bin/bash

# Start FastAPI backend on port 8000 (internal)
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Wait for backend to be ready
sleep 5

# Start Streamlit frontend on Render's assigned PORT
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
