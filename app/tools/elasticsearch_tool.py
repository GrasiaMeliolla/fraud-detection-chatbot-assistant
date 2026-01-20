"""
Elasticsearch tool for hybrid search (semantic + keyword).
Pattern adapted from bappeda agentic_system.
"""
from typing import List, Dict, Optional
from elasticsearch import AsyncElasticsearch
from app.config import settings
from app.services.embedding_service import embedding_service
import logging

logger = logging.getLogger(__name__)


class ElasticsearchTool:
    """Tool for hybrid search using Elasticsearch."""

    def __init__(self):
        self.client = None
        self.index_name = settings.elasticsearch_index_name

    async def _get_client(self) -> AsyncElasticsearch:
        """Get or create Elasticsearch client."""
        if self.client is None:
            self.client = AsyncElasticsearch([settings.elasticsearch_url])
        return self.client

    async def create_index(self):
        """Create index with proper mapping for vector search."""
        client = await self._get_client()

        # Check if index exists
        exists = await client.indices.exists(index=self.index_name)
        if exists:
            logger.info(f"Index {self.index_name} already exists")
            return

        # Create index with mapping
        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1536,  # text-embedding-3-small dimension
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "properties": {
                            "filename": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk_id": {"type": "integer"},
                            "total_chunks": {"type": "integer"}
                        }
                    }
                }
            }
        }

        await client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created index {self.index_name}")

    async def index_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict,
        doc_id: Optional[str] = None
    ):
        """Index a single document chunk."""
        client = await self._get_client()

        body = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }

        if doc_id:
            await client.index(index=self.index_name, id=doc_id, body=body)
        else:
            await client.index(index=self.index_name, body=body)

        logger.debug(f"Indexed document: {doc_id or 'auto-id'}")

    async def index_batch(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict]
    ):
        """Index multiple documents in batch."""
        client = await self._get_client()

        actions = []
        for chunk, embedding, metadata in zip(chunks, embeddings, metadata_list):
            actions.append({"index": {"_index": self.index_name}})
            actions.append({
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata
            })

        if actions:
            await client.bulk(operations=actions)
            logger.info(f"Indexed {len(chunks)} documents in batch")

    async def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query text
            query_embedding: Pre-computed embedding (optional, will generate if not provided)
            top_k: Number of results to return
            filters: Additional Elasticsearch filters

        Returns:
            List of search results with text, metadata, and score
        """
        client = await self._get_client()

        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = await embedding_service.embed(query)

        # Build hybrid search query
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # Semantic search using embeddings (70% weight)
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": 0.7
                            }
                        },
                        # Keyword search (30% weight)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^3", "metadata.filename^2"],
                                "fuzziness": "AUTO",
                                "boost": 0.3
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        # Add filters if provided
        if filters:
            search_body["query"]["bool"]["must"] = filters

        try:
            response = await client.search(index=self.index_name, body=search_body)

            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"]
                })

            logger.info(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def delete_index(self):
        """Delete the index."""
        client = await self._get_client()
        if await client.indices.exists(index=self.index_name):
            await client.indices.delete(index=self.index_name)
            logger.info(f"Deleted index {self.index_name}")

    async def close(self):
        """Close Elasticsearch connection."""
        if self.client:
            await self.client.close()
            logger.info("Closed Elasticsearch connection")


# Singleton instance
elasticsearch_tool = ElasticsearchTool()
