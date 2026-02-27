"""Linear LLM pipeline: first step is extractor (structured extraction from text)."""
from .extractor import extract

__all__ = ["extract"]
