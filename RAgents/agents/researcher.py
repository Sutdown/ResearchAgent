from typing import Dict, List, Optional
from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader
from RAgents.tools.arxiv_search import ArxivSearch
from RAgents.tools.mcp_client import MCPClient
from RAgents.tools.tavily_search import TavilySearch


class Researcher:
    def __init__(
            self,
            llm: BaseLLM,
            tavily_api_key: Optional[str] = None,
            mcp_server_url: Optional[str] = None,
            mcp_api_key: Optional[str] = None,
            enable_vector_memory: bool = True,
            vector_memory_path: str = "./vector_memory"
    ):
        # llm，tools，prompt
        self.llm = llm
        self.tavily = TavilySearch(tavily_api_key) if tavily_api_key else None
        self.arxiv = ArxivSearch()
        self.mcp = MCPClient(mcp_server_url, mcp_api_key) if mcp_server_url else None
        self.prompt_loader = PromptLoader()
        self.max_requests_per_task: int = 4
        # vetctor memory
        self.enable_vector_memory = enable_vector_memory
        if enable_vector_memory:
            self.vector_memory = VectorMemory(persist_directory=vector_memory_path)
        else:
            self.vector_memory = None