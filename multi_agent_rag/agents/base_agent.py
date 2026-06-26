from dataclasses import dataclass, field
from typing import Any
import anthropic
from config import CLAUDE_MODEL, MAX_TOKENS, ANTHROPIC_API_KEY


@dataclass
class AgentMessage:
    role: str          # "user" | "assistant"
    content: str
    metadata: dict = field(default_factory=dict)


class BaseAgent:
    """Base class for all agents in the multi-agent RAG system."""

    name: str = "BaseAgent"
    description: str = ""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.history: list[AgentMessage] = []

    def _call_claude(self, system: str, messages: list[dict], use_thinking: bool = True) -> str:
        kwargs: dict[str, Any] = {
            "model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        # Extract text from response content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def run(self, input_data: Any) -> Any:
        raise NotImplementedError
