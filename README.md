# Groq RAG Chatbot ⚡

A real-time, streaming AI chatbot powered by **Groq LLM** with full **RAG** (Retrieval-Augmented Generation) capabilities.

## Features

- ⚡ **Groq LLM** — LLaMA 3.3 70B with blazing-fast streaming
- 🖼️ **Vision AI** — Image understanding via LLaMA 3.2 Vision
- 📄 **Multi-format Upload** — PDF, DOCX, TXT, CSV, JPG, PNG
- 🔍 **OCR** — Extract text from images using EasyOCR
- 🌐 **Web Search** — Real-time internet search via Tavily
- 🧠 **Smart Routing** — Auto-detects if query needs docs, web, or both
- 💬 **Conversation Memory** — Remembers chat context
- 📎 **Source Citations** — Shows where answers come from

## Setup

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) |
| Vision | Groq (LLaMA 3.2 90B Vision) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Web Search | Tavily API |
| OCR | EasyOCR |
| Frontend | Streamlit |

## License

MIT
