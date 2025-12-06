from typing import Optional
from RAgents.llms.base import BaseLLM

class LLMFactory:
    _providers = {}

    @classmethod
    def register_provider(cls, name: str, provider_class):
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create_llm(
            cls,
            provider: str,
            api_key: str,
            model: Optional[str] = None,
            **kwargs
    ) -> BaseLLM:
        provider = provider.lower()
        # 调用到特定的llm模块时才会import，减少导入时间和资源消耗
        if provider not in cls._providers:
            cls._lazy_load_provider(provider)

        if provider not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Available providers: {available}"
            )

        llm_class = cls._providers[provider]
        if model:
            return llm_class(api_key=api_key, model=model, **kwargs)
        else:
            return llm_class(api_key=api_key, **kwargs)

    @classmethod
    def _lazy_load_provider(cls, provider: str):
        try:
            if provider == 'deepseek':
                from RAgents.llms.deepseek import DeepSeekLLM
                cls.register_provider('deepseek', DeepSeekLLM)
        except ImportError as e:
            pass

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._providers.keys())