import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parent / ".env")


def build_llm() -> ChatGroq:
    """Build the shared chat model from the LLM_* variables in .env."""

    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise ValueError("LLM_API_KEY is missing. Please add it to your .env file.")

    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    base_url = os.getenv("LLM_BASE_URL") or None
    max_rpm = float(os.getenv("LLM_MAX_RPM", "30"))

    return ChatGroq(
        model=model,
        api_key=api_key,
        base_url=base_url,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=max_rpm / 60,
            max_bucket_size=1,
        ),
    )
