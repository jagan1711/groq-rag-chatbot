"""
RAG Engine Module.
Orchestrates the full Retrieval-Augmented Generation pipeline:
document processing → chunking → embedding → retrieval → context building.
"""

import logging
from typing import Generator, Optional

from document_processor import DocumentProcessor
from chunker import RecursiveChunker
from vector_store import VectorStore
from llm_client import LLMClient
from web_search import WebSearcher
from router import QueryRouter
from memory import ConversationMemory

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine that ties together all components:
    document ingestion, retrieval, web search, routing, and generation.
    """

    def __init__(self):
        """Initialize all RAG components."""
        self.doc_processor = DocumentProcessor()
        self.chunker = RecursiveChunker()
        self.vector_store = VectorStore()
        self.llm = LLMClient()
        self.web_searcher = WebSearcher()
        self.router = QueryRouter(llm_client=self.llm)
        self.memory = ConversationMemory()

        logger.info("RAG Engine initialized with all components.")

    # ─── Document Ingestion ──────────────────────────────────────────────

    def ingest_file(self, file_name: str, file_bytes: bytes) -> dict:
        """
        Process and store a file in the vector store.

        Args:
            file_name: Original filename.
            file_bytes: Raw file content.

        Returns:
            dict with 'source', 'chunks_stored', 'type', 'vision_analysis'.
        """
        result = {
            "source": file_name,
            "chunks_stored": 0,
            "type": "",
            "vision_analysis": None,
        }

        # 1. Extract text
        doc = self.doc_processor.process_file(file_name, file_bytes)
        result["type"] = doc["type"]
        text = doc["text"]

        # 2. If image, also run Vision analysis for richer understanding
        if doc["is_image"]:
            try:
                img_b64 = self.doc_processor.get_image_base64(file_bytes)
                vision_text = self.llm.analyze_image(img_b64)
                result["vision_analysis"] = vision_text
                # Combine OCR text with vision analysis
                text = f"[OCR Text]: {text}\n\n[Visual Analysis]: {vision_text}"
            except Exception as e:
                logger.warning(f"Vision analysis skipped: {e}")

        # 3. Chunk the text
        chunks = self.chunker.chunk_text(text, source=file_name)

        # 4. Store in vector DB
        if chunks:
            stored = self.vector_store.add_documents(chunks)
            result["chunks_stored"] = stored

        logger.info(
            f"Ingested '{file_name}': {result['chunks_stored']} chunks stored."
        )
        return result

    # ─── Query Processing ────────────────────────────────────────────────

    def query(self, user_message: str) -> Generator[str, None, None]:
        """
        Process a user query through the full RAG pipeline with streaming.

        Args:
            user_message: The user's question.

        Yields:
            Response tokens (streamed).
        """
        has_docs = self.vector_store.count > 0

        # 1. Route the query
        route = self.router.route(user_message, has_docs)
        logger.info(f"Query routed as: {route}")

        # 2. Gather context based on route
        doc_context = ""
        web_context = ""

        if route in ("DOCS_ONLY", "BOTH"):
            doc_results = self.vector_store.search(user_message)
            if doc_results:
                doc_context = self._format_doc_context(doc_results)

        if route in ("WEB_ONLY", "BOTH"):
            web_results = self.web_searcher.search(user_message)
            if web_results:
                web_context = self.web_searcher.format_results(web_results)

        # 3. If DOCS_ONLY route found no docs, fall back to web
        if route == "DOCS_ONLY" and not doc_context:
            logger.info("No document results found. Falling back to web search.")
            web_results = self.web_searcher.search(user_message)
            if web_results:
                web_context = self.web_searcher.format_results(web_results)

        # 4. Add user message to memory
        self.memory.add_user_message(user_message)

        # 5. Stream response from LLM
        full_response = ""
        for token in self.llm.stream_chat(
            user_message=user_message,
            context=doc_context,
            web_results=web_context,
            chat_history=self.memory.get_history()[:-1],  # Exclude current msg
        ):
            full_response += token
            yield token

        # 6. Store assistant response in memory
        self.memory.add_assistant_message(full_response)

    # ─── Utility Methods ─────────────────────────────────────────────────

    def get_sources(self) -> list[str]:
        """Return list of all document sources in the vector store."""
        return self.vector_store.get_document_sources()

    def delete_source(self, source: str) -> int:
        """Delete a document and its chunks from the store."""
        return self.vector_store.delete_by_source(source)

    def clear_all(self) -> None:
        """Clear all documents and conversation history."""
        self.vector_store.clear()
        self.memory.clear()
        logger.info("All data cleared (documents + memory).")

    @property
    def document_count(self) -> int:
        """Total number of chunks in the vector store."""
        return self.vector_store.count

    # ─── Private Methods ─────────────────────────────────────────────────

    def _format_doc_context(self, results: list[dict]) -> str:
        """Format retrieved document chunks into context for the LLM."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"**[Chunk {i}]** (Source: {result['source']}, "
                f"Relevance: {result['relevance_score']:.0%})\n"
                f"{result['text']}"
            )
        return "\n\n---\n\n".join(context_parts)
