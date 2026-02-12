"""
LLM Client Module.
Handles all interactions with Groq's API for both text generation
(with streaming) and vision-based image analysis.
"""

import base64
import logging
from typing import Generator, Optional

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, VISION_MODEL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMClient:
    """Groq LLM client for streaming text generation and vision analysis."""

    def __init__(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Check your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"LLM Client initialized (model: {LLM_MODEL}).")

    def stream_chat(
        self,
        user_message: str,
        context: str = "",
        web_results: str = "",
        chat_history: list[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat response from Groq, augmented with RAG context.

        Args:
            user_message: The user's query.
            context: Retrieved document context (from RAG).
            web_results: Formatted web search results.
            chat_history: Previous conversation messages.

        Yields:
            Response tokens one at a time.
        """
        messages = self._build_messages(
            user_message, context, web_results, chat_history
        )

        try:
            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            yield f"\n\n⚠️ Error generating response: {str(e)}"

    def generate_response(
        self,
        user_message: str,
        context: str = "",
        web_results: str = "",
        chat_history: list[dict] = None,
    ) -> str:
        """
        Generate a complete (non-streaming) response.

        Args:
            user_message: The user's query.
            context: Retrieved document context.
            web_results: Formatted web search results.
            chat_history: Previous conversation messages.

        Returns:
            Complete response string.
        """
        messages = self._build_messages(
            user_message, context, web_results, chat_history
        )

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return f"⚠️ Error generating response: {str(e)}"

    def analyze_image(
        self,
        image_base64: str,
        prompt: str = "Describe this image in detail. Extract all text, data, diagrams, and visual information.",
    ) -> str:
        """
        Analyze an image using Groq's Vision model.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Instruction for the vision model.

        Returns:
            Text description / analysis of the image.
        """
        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            result = response.choices[0].message.content
            logger.info(f"Vision analysis complete: {len(result)} chars.")
            return result
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"[Vision analysis failed: {str(e)}]"

    def classify_query(self, query: str, has_documents: bool) -> str:
        """
        Use the LLM to classify whether a query needs docs, web, or both.

        Args:
            query: The user's query.
            has_documents: Whether documents are loaded.

        Returns:
            One of: 'DOCS_ONLY', 'WEB_ONLY', 'BOTH', 'GENERAL'.
        """
        classification_prompt = f"""Classify this user query into exactly one category.
Available categories:
- DOCS_ONLY: The user is asking about uploaded documents or files.
- WEB_ONLY: The user is asking about current events, news, or real-time information.
- BOTH: The user wants to compare document info with web info.
- GENERAL: General conversation, greetings, or questions not needing search.

User has documents loaded: {has_documents}

Query: "{query}"

Respond with ONLY the category name (e.g., DOCS_ONLY). Nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            category = response.choices[0].message.content.strip().upper()
            if category in ("DOCS_ONLY", "WEB_ONLY", "BOTH", "GENERAL"):
                return category
            return "GENERAL"
        except Exception:
            return "GENERAL"

    # ─── Private Methods ─────────────────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        context: str,
        web_results: str,
        chat_history: Optional[list[dict]],
    ) -> list[dict]:
        """Build the full message payload for the Groq API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        if chat_history:
            messages.extend(chat_history)

        # Build the augmented user message
        augmented_parts = []

        if context:
            augmented_parts.append(
                f"### 📄 Relevant Document Context:\n{context}"
            )

        if web_results:
            augmented_parts.append(
                f"### 🌐 Web Search Results:\n{web_results}"
            )

        if augmented_parts:
            augmented_msg = (
                "\n\n".join(augmented_parts)
                + f"\n\n### 💬 User Question:\n{user_message}"
            )
        else:
            augmented_msg = user_message

        messages.append({"role": "user", "content": augmented_msg})
        return messages
