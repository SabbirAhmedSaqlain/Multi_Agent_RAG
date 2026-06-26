from typing import Any
import anthropic
from config import CLAUDE_MODEL, MAX_TOKENS, ANTHROPIC_API_KEY
from utils import get_logger


class BaseAgent:
    name: str = "BaseAgent"

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.log = get_logger(self.name)

    def _call(self, system: str, user_msg: str, thinking: bool = True) -> str:
        self.log.debug("Calling Claude (%s tokens budget)", MAX_TOKENS)
        kwargs: dict[str, Any] = {
            "model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(parts).strip()
