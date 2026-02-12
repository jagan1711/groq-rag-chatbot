"""
Configuration module for the Groq RAG Chatbot.
Loads environment variables and defines all application constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ─────────────────────────────────────────────
load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ─── Model Configuration ────────────────────────────────────────────────────
LLM_MODEL: str = "llama-3.3-70b-versatile"
VISION_MODEL: str = "llama-3.2-90b-vision-preview"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─── Chunking Parameters ────────────────────────────────────────────────────
CHUNK_SIZE: int = 500          # Characters per chunk
CHUNK_OVERLAP: int = 50        # Overlap between consecutive chunks

# ─── Retrieval Settings ─────────────────────────────────────────────────────
TOP_K_RESULTS: int = 5         # Number of similar chunks to retrieve
SIMILARITY_THRESHOLD: float = 0.3  # Minimum similarity score to include

# ─── Conversation Memory ────────────────────────────────────────────────────
MAX_MEMORY_MESSAGES: int = 20  # Max message pairs to keep in memory

# ─── Web Search ──────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = 5    # Max web search results to fetch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")

# ─── Supported File Extensions ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".txt": "Text File",
    ".csv": "CSV File",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".png": "PNG Image",
}

# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are an intelligent AI assistant with access to the user's uploaded documents and web search capabilities.

Your behavior:
1. When the user asks about their uploaded documents, answer ONLY from the provided context. Cite the source document name.
2. When the user asks about current events or general knowledge not in documents, use web search results. Cite the source URL.
3. When both document context and web results are provided, synthesize a comprehensive answer citing both.
4. If you don't have enough context to answer, say so honestly. Never fabricate information.
5. Keep responses clear, well-structured, and helpful. Use markdown formatting.
6. When citing sources, use this format:
   - Document sources: 📄 *Source: [filename]*
   - Web sources: 🌐 *Source: [title](URL)*

Remember: Accuracy and helpfulness are your top priorities."""
