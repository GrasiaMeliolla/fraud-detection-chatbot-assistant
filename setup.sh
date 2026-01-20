#!/bin/bash

# Fina Chatbot Complete Setup Script (Docker Compose)

set -e

echo "================================================"
echo "Fina - Fraud Detection Chatbot Setup"
echo "================================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and add your OPENAI_API_KEY"
    exit 1
fi

echo "✓ Environment file found"

# Check if data files exist
if [ ! -f data/fraudTrain.csv ] || [ ! -f data/fraudTest.csv ]; then
    echo ""
    echo "WARNING: CSV files not found in data/"
    echo "Please place fraudTrain.csv and fraudTest.csv in data/ directory"
    echo ""
fi

PDF_COUNT=$(find data/documents -name "*.pdf" 2>/dev/null | wc -l)
if [ $PDF_COUNT -eq 0 ]; then
    echo "WARNING: No PDF files found in data/documents/"
    echo "Please place fraud detection PDF files in data/documents/"
    echo ""
fi

# Build and start all Docker services
echo "Building and starting all services..."
echo "(This may take a few minutes on first run)"
docker-compose up -d --build

echo ""
echo "Waiting for services to be healthy..."
sleep 20

echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "================================================"
echo "Services started successfully!"
echo "================================================"

# Check if data files exist and ingest
if [ -f data/fraudTrain.csv ] && [ -f data/fraudTest.csv ]; then
    echo ""
    echo "Ingesting transaction data..."
    docker-compose exec -T fastapi python scripts/ingest_data.py
    echo "✓ Data ingested successfully"
else
    echo ""
    echo "Skipping data ingestion (CSV files not found)"
    echo "To ingest later: docker-compose exec fastapi python scripts/ingest_data.py"
fi

# Check if PDF documents exist and process
if [ $PDF_COUNT -gt 0 ]; then
    echo ""
    echo "Processing PDF documents..."
    docker-compose exec -T fastapi python scripts/process_documents.py
    echo "✓ Documents processed successfully"
else
    echo ""
    echo "Skipping document processing (PDF files not found)"
    echo "To process later: docker-compose exec fastapi python scripts/process_documents.py"
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Fina is now running at:"
echo "  - UI:       http://localhost:8501"
echo "  - API:      http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f fastapi"
echo "  docker-compose logs -f streamlit"
echo ""
echo "To stop:"
echo "  docker-compose down"
echo "================================================"
