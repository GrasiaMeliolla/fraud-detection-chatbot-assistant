#!/bin/bash

# Fina Chatbot Startup Script

set -e

echo "================================================"
echo "Starting Fina - Fraud Detection Chatbot"
echo "================================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if data files exist
if [ ! -f data/fraudTrain.csv ] || [ ! -f data/fraudTest.csv ]; then
    echo "WARNING: CSV files not found in data/"
    echo "Please place fraudTrain.csv and fraudTest.csv in the data/ directory"
    echo ""
fi

# Check if PDF documents exist
if [ ! -f data/documents/*.pdf ]; then
    echo "WARNING: No PDF files found in data/documents/"
    echo "Please place fraud detection PDF files in data/documents/"
    echo ""
fi

# Start Docker services
echo "Starting Docker services (PostgreSQL, Elasticsearch, Redis)..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

# Check if services are running
echo "Checking service health..."
docker-compose ps

echo ""
echo "================================================"
echo "Services started successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Ingest data: python scripts/ingest_data.py"
echo "2. Process documents: python scripts/process_documents.py"
echo "3. Start API: python -m uvicorn app.main:app --reload"
echo "4. Start UI: streamlit run ui/streamlit_app.py"
echo ""
echo "Or run the setup script: ./setup.sh"
echo "================================================"
