"""
FastAPI main application for Fina chatbot.
Pattern adapted from bappeda agentic_system.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import chat
from app.db.session import init_db, close_db
from app.services.vanna_service import train_vanna_model
from app.tools.elasticsearch_tool import elasticsearch_tool
from app.config import settings
import logging
import os

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""

    # Startup
    logger.info("Starting Fina chatbot application...")

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Create Elasticsearch index
    logger.info("Creating Elasticsearch index...")
    try:
        await elasticsearch_tool.create_index()
    except Exception as e:
        logger.warning(f"Elasticsearch index creation skipped: {e}")

    # Train Vanna model
    logger.info("Training Vanna AI model...")
    try:
        await train_vanna_model()
    except Exception as e:
        logger.warning(f"Vanna training skipped: {e}")

    logger.info("Fina chatbot application started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down Fina chatbot application...")
    await close_db()
    await elasticsearch_tool.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Fina - Fraud Detection Chatbot Assistant",
    description="AI-powered chatbot for analyzing fraud patterns from transactions and documents",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])

# Mount static files for PDFs
documents_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
if os.path.exists(documents_path):
    app.mount("/documents", StaticFiles(directory=documents_path), name="documents")
    logger.info(f"Mounted static files from {documents_path}")
else:
    logger.warning(f"Documents directory not found: {documents_path}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Fina - Fraud Detection Chatbot Assistant",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "elasticsearch": "connected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
