from typing import Iterator, Any
from openai import OpenAI, APIConnectionError
from RAgents.llms.base import BaseLLM


class DeepSeekLLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        params = {**self.config, **kwargs}
        params.setdefault("timeout", 60)
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **params,
                )
                return response.choices[0].message.content
            except APIConnectionError as e:
                # Save error and retry once; if it still fails, re-raise below.
                last_error = e
                continue
        # If we get here, all retries failed.
        assert last_error is not None
        raise last_error

    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        params = {**self.config, **kwargs}
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **params
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content