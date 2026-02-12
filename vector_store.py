"""
Vector Store Module.
Manages ChromaDB for persistent storage and retrieval of document embeddings.
"""

import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from embeddings import get_embeddings, get_single_embedding

logger = logging.getLogger(__name__)

# ─── Collection Name ─────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_documents"


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: str = CHROMA_DB_DIR):
        """
        Initialize the ChromaDB client and collection.

        Args:
            persist_dir: Directory for persistent ChromaDB storage.
        """
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore initialized. Collection '{COLLECTION_NAME}' "
            f"has {self.collection.count()} documents."
        )

    def add_documents(
        self,
        chunks: list[dict],
    ) -> int:
        """
        Embed and store document chunks in the vector store.

        Args:
            chunks: List of dicts with keys 'text', 'source', 'chunk_index'.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
            for chunk in chunks
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Generate embeddings
        embeddings = get_embeddings(texts)

        # Upsert into ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"Stored {len(chunks)} chunks from "
            f"'{chunks[0]['source']}' into vector store."
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Search for the most relevant document chunks.

        Args:
            query: The user's search query.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: 'text', 'source', 'chunk_index',
            'distance', 'relevance_score'.
        """
        if self.collection.count() == 0:
            logger.info("Vector store is empty. No results.")
            return []

        query_embedding = get_single_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        search_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            relevance = 1.0 - (distance / 2.0)

            if relevance >= SIMILARITY_THRESHOLD:
                search_results.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "distance": distance,
                    "relevance_score": round(relevance, 4),
                })

        # Sort by relevance (highest first)
        search_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info(
            f"Search for '{query[:50]}...' returned "
            f"{len(search_results)} relevant results."
        )
        return search_results

    def get_document_sources(self) -> list[str]:
        """Return a list of unique source document names in the store."""
        if self.collection.count() == 0:
            return []

        all_data = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data["metadatas"]:
            sources.add(meta.get("source", "unknown"))
        return sorted(sources)

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a specific source document.

        Args:
            source: The source filename to delete.

        Returns:
            Number of chunks deleted.
        """
        all_data = self.collection.get(
            include=["metadatas"],
            where={"source": source},
        )
        ids_to_delete = all_data["ids"]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks from source '{source}'.")
        return len(ids_to_delete)

    def clear(self) -> None:
        """Delete all documents from the vector store."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")

    @property
    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self.collection.count()
