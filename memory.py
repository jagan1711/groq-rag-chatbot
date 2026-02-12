"""
Conversation Memory Module.
Manages chat history with a sliding window to stay within
LLM context limits while maintaining conversation coherence.
"""

import logging
from typing import Optional

from config import MAX_MEMORY_MESSAGES

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Sliding-window conversation memory.
    Stores the most recent N message pairs (user + assistant).
    """

    def __init__(self, max_messages: int = MAX_MEMORY_MESSAGES):
        """
        Args:
            max_messages: Maximum number of individual messages to retain.
        """
        self.max_messages = max_messages
        self._history: list[dict] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to history."""
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        """
        Return the current conversation history.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return list(self._history)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        logger.info("Conversation memory cleared.")

    @property
    def message_count(self) -> int:
        """Return the number of messages in memory."""
        return len(self._history)

    def _trim(self) -> None:
        """Remove oldest messages if exceeding the limit."""
        if len(self._history) > self.max_messages:
            excess = len(self._history) - self.max_messages
            self._history = self._history[excess:]
            logger.debug(f"Memory trimmed: removed {excess} oldest messages.")
