"""Small numerical references, not a DeepSeek model implementation."""

from .engine import Engine, EngineConfig, RequestError, Result
from .model import TinyModel

__version__ = "0.1.0"
__all__ = ["Engine", "EngineConfig", "RequestError", "Result", "TinyModel"]
