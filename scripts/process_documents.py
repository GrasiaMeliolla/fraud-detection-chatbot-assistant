"""
Document processing script to load PDFs into Elasticsearch.
Pattern adapted from bappeda agentic_system.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from typing import List
import PyPDF2
from app.services.embedding_service import embedding_service
from app.tools.elasticsearch_tool import elasticsearch_tool
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf_with_pages(pdf_path: str) -> List[dict]:
    """
    Extract text from PDF file with page numbers.

    Returns:
        List of dicts with {'page': int, 'text': str}
    """
    logger.info(f"Extracting text from {pdf_path}...")

    pages_data = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_data.append({
                    'page': page_num + 1,  # 1-indexed for human readability
                    'text': text
                })

    logger.info(f"Extracted {len(pages_data)} pages from {num_pages} total pages")
    return pages_data


def chunk_text_with_pages(pages_data: List[dict], chunk_size: int = 500, overlap: int = 100) -> List[dict]:
    """
    Split text into chunks with page tracking.

    Args:
        pages_data: List of {'page': int, 'text': str}
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of {'text': str, 'page': int, 'chunk_id': int}
    """
    chunks = []
    chunk_id = 0

    for page_data in pages_data:
        page_num = page_data['page']
        text = page_data['text']
        words = text.split()

        # If page is small, keep it as one chunk
        if len(words) <= chunk_size:
            chunks.append({
                'text': text,
                'page': page_num,
                'chunk_id': chunk_id
            })
            chunk_id += 1
        else:
            # Split large pages into chunks
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i:i + chunk_size])
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'page': page_num,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1

    logger.info(f"Created {len(chunks)} chunks from {len(pages_data)} pages")
    return chunks


async def process_pdf(pdf_path: str, filename: str):
    """Process a single PDF file and index to Elasticsearch."""

    logger.info("=" * 60)
    logger.info(f"Processing: {filename}")
    logger.info("=" * 60)

    # Extract text with page numbers
    pages_data = extract_text_from_pdf_with_pages(pdf_path)

    # Chunk text with page tracking
    chunks_data = chunk_text_with_pages(pages_data, chunk_size=500, overlap=100)
    logger.info(f"Total chunks: {len(chunks_data)}")

    # Extract text and metadata
    chunk_texts = [chunk['text'] for chunk in chunks_data]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = await embedding_service.embed_batch(chunk_texts)

    # Prepare metadata with page numbers
    metadata_list = []
    for chunk in chunks_data:
        metadata_list.append({
            "filename": filename,
            "page": chunk['page'],  # NOW INCLUDING PAGE NUMBER
            "chunk_id": chunk['chunk_id'],
            "total_chunks": len(chunks_data)
        })

    # Index to Elasticsearch
    logger.info("Indexing to Elasticsearch...")
    await elasticsearch_tool.index_batch(chunk_texts, embeddings, metadata_list)

    logger.info(f"Successfully processed {filename}")


async def main():
    """Main processing function."""

    # Create Elasticsearch index
    logger.info("Creating Elasticsearch index...")
    await elasticsearch_tool.create_index()

    # Find PDF files in documents directory
    docs_dir = Path("data/documents")

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        logger.info("Please create data/documents/ and place PDF files there")
        return

    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in data/documents/")
        logger.info("Please place fraud detection PDF files in data/documents/")
        return

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            await process_pdf(str(pdf_file), pdf_file.name)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            continue

    logger.info("=" * 60)
    logger.info("Document processing complete!")
    logger.info("=" * 60)

    # Close Elasticsearch connection
    await elasticsearch_tool.close()


if __name__ == "__main__":
    asyncio.run(main())
