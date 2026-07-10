from llm import chat
from utils import get_logger


class BaseAgent:
    """All agents talk to the LLM through the provider-agnostic `llm.chat`,
    so the whole pipeline runs identically on Anthropic, Ollama, LM Studio,
    or any OpenAI-compatible endpoint (see config.LLM_PROVIDER)."""

    name: str = "BaseAgent"

    def __init__(self):
        self.log = get_logger(self.name)

    def _call(self, system: str, user_msg: str, thinking: bool = True) -> str:
        return chat(system, user_msg, thinking=thinking)
