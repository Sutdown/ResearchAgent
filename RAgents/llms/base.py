from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


class BaseLLM(ABC):
    def __init__(self, api_key: str, model: str, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self.config: Dict[str, Any] = dict(kwargs)

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"




