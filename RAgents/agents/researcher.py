from typing import Dict, List, Optional
from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader
from RAgents.tools.arxiv_search import ArxivSearch
from RAgents.tools.mcp_client import MCPClient
from RAgents.tools.tavily_search import TavilySearch
from RAgents.utils.vector import VectorMemory
from RAgents.workflow.state import ResearchState, SubTask, SearchResult


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
        self.max_requests_per_task: int = 3
        # vetctor memory
        self.enable_vector_memory = enable_vector_memory
        if enable_vector_memory:
            self.vector_memory = VectorMemory(persist_directory=vector_memory_path)
        else:
            self.vector_memory = None

    # 执行单个任务
    def execute_task(self, state: ResearchState, task: SubTask) -> ResearchState:
        # 从向量库中获取历史结果，满足条件的情况下可以缓存结果重用，直接完成任务
        if self.vector_memory:
            task_description = task.get("description", '')
            similar_queries = self.vector_memory.find_similar_queries(
                task_description,
                threshold=0.8,
                limit=3
            )
            high_quality_similar = [
                q for q in similar_queries
                if q['similarity'] > 0.9 and q['quality_score'] >= 4.0
            ]
            if high_quality_similar:
                best_similar = max(high_quality_similar, key=lambda x: x['similarity'])
                results = self._create_cached_result(task, best_similar) # 转换最相似的结果直接返回
                self._add_results_to_state(state, results, task) # 将结果加入状态中，标记当前任务完成
                return state

        # 标准搜索流程，根据任务中的搜索引擎和搜索问题进行搜索
        results = []
        request_count = 0
        for query in task.get('search_queries', []):
            for source in task.get('sources', []): # 查询源头（限制1）
                if request_count >= self.max_requests_per_task: # 最大查询次数（限制2）
                    break
                result = self._search(query, source)
                request_count += 1
                if result:
                    result['task_id'] = task['task_id']
                    results.append(result)
            if request_count >= self.max_requests_per_task:
                break

        # 将结果存储到向量内存
        if self.vector_memory and results:
            task_description = task.get('description', '')
            self.vector_memory.store_research_result(
                query=task_description,
                results={'search_results': results},
                quality_score=0.0,  # Will be updated based on user feedback
                metadata={
                    'task_id': task['task_id'],
                    'sources_used': task.get('sources', []),
                    'queries_used': task.get('search_queries', [])
                }
            )

        # 整合结果
        self._add_results_to_state(state, results, task)
        return state

    def _search(self, query: str, source: str) -> Optional[SearchResult]:
        try:
            if source == 'tavily' and self.tavily:
                return self.tavily.search(query)
            elif source == 'arxiv':
                return self.arxiv.search(query)
            elif source == 'mcp' and self.mcp:
                import asyncio
                return asyncio.run(self.mcp.search(query))
            else:
                return None
        except Exception as e:
            return {
                'query': query,
                'source': source,
                'results': [],
                'error': str(e)
            }

    def _create_cached_result(self, task: SubTask, similar_query: Dict) -> List[SearchResult]:
        """Create result from cached similar query."""
        return [{
            'query': task.get('description', ''),
            'source': 'cached_vector_memory',
            'results': similar_query.get('results_summary', ''),
            'task_id': task['task_id'],
            'similarity_score': similar_query['similarity'],
            'cached_quality': similar_query['quality_score'],
            'cache_timestamp': similar_query['timestamp']
        }]

    def _add_results_to_state(self, state: ResearchState, results: List[SearchResult], task: SubTask):
        # 将结果添加到状态中
        if 'research_results' not in state:
            state['research_results'] = []
        state['research_results'].extend(results)

        # 标记当前任务为已完成
        if state.get('research_plan'):
            for t in state['research_plan'].get('sub_tasks', []):
                if t.get('task_id') == task['task_id']:
                    t['status'] = 'completed'
                    break