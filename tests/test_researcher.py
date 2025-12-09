import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from RAgents.agents.researcher import Researcher
from RAgents.llms.base import BaseLLM
from RAgents.workflow.state import ResearchState


class MockLLM(BaseLLM):
    """用于测试的模拟 LLM 类"""
    
    def __init__(self, api_key: str = "test_key", model: str = "test_model", **kwargs):
        super().__init__(api_key, model, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        return "Test response"
    
    def stream_generate(self, prompt: str, **kwargs):
        yield "Test response"


class TestResearcher:
    """测试 Researcher 类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.mock_llm = MockLLM()
        
    @patch('RAgents.agents.researcher.TavilySearch')
    @patch('RAgents.agents.researcher.ArxivSearch')
    @patch('RAgents.agents.researcher.MCPClient')
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_init_with_all_options(self, mock_prompt_loader, mock_vector_memory, 
                                   mock_mcp_client, mock_arxiv_search, mock_tavily_search):
        """测试完整配置的初始化"""
        mock_tavily_search.return_value = Mock()
        mock_arxiv_search.return_value = Mock()
        mock_mcp_client.return_value = Mock()
        mock_vector_memory.return_value = Mock()
        
        researcher = Researcher(
            llm=self.mock_llm,
            tavily_api_key="tavily_key",
            mcp_server_url="http://server",
            mcp_api_key="mcp_key",
            enable_vector_memory=True,
            vector_memory_path="./test_memory"
        )
        
        assert researcher.llm == self.mock_llm
        assert researcher.tavily is not None
        assert researcher.arxiv is not None
        assert researcher.mcp is not None
        assert researcher.vector_memory is not None
        assert researcher.max_requests_per_task == 3
    
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_init_without_optional_components(self, mock_prompt_loader, mock_vector_memory):
        """测试不包含可选组件的初始化"""
        researcher = Researcher(
            llm=self.mock_llm,
            tavily_api_key=None,
            mcp_server_url=None,
            enable_vector_memory=False
        )
        
        assert researcher.llm == self.mock_llm
        assert researcher.tavily is None
        assert researcher.mcp is None
        assert researcher.vector_memory is None
    
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_execute_task_with_vector_cache_hit(self, mock_prompt_loader, mock_vector_memory):
        """测试使用向量缓存命中执行任务"""
        # 设置模拟向量内存
        mock_memory_instance = Mock()
        mock_vector_memory.return_value = mock_memory_instance
        
        # 设置高质量缓存命中
        mock_similar_query = {
            'similarity': 0.95,
            'quality_score': 4.5,
            'results_summary': 'Cached results',
            'timestamp': '2023-01-01T00:00:00'
        }
        mock_memory_instance.find_similar_queries.return_value = [mock_similar_query]
        
        researcher = Researcher(llm=self.mock_llm, enable_vector_memory=True)
        
        # 创建测试任务和状态
        task = {
            'task_id': 1,
            'description': '研究人工智能发展',
            'search_queries': ['AI development'],
            'sources': ['tavily']
        }
        state = self._create_test_state()
        
        # 执行任务
        updated_state = researcher.execute_task(state, task)
        
        # 验证结果
        assert len(updated_state['research_results']) == 1
        result = updated_state['research_results'][0]
        assert result['source'] == 'cached_vector_memory'
        assert result['similarity_score'] == 0.95
        assert result['cached_quality'] == 4.5
    
    @patch('RAgents.agents.researcher.TavilySearch')
    @patch('RAgents.agents.researcher.ArxivSearch')
    @patch('RAgents.agents.researcher.MCPClient')
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_execute_task_standard_search(self, mock_prompt_loader, mock_vector_memory,
                                          mock_mcp_client, mock_arxiv_search, mock_tavily_search):
        """测试标准搜索流程"""
        # 设置搜索工具
        mock_tavily_instance = Mock()
        mock_arxiv_instance = Mock()
        mock_memory_instance = Mock()
        
        mock_tavily_search.return_value = mock_tavily_instance
        mock_arxiv_search.return_value = mock_arxiv_instance
        mock_vector_memory.return_value = mock_memory_instance
        
        # 设置搜索结果
        mock_tavily_instance.search.return_value = {
            'query': 'test query',
            'source': 'tavily',
            'results': [{'title': 'Result 1', 'content': 'Content 1'}]
        }
        mock_memory_instance.find_similar_queries.return_value = []  # 无缓存命中
        
        researcher = Researcher(
            llm=self.mock_llm,
            tavily_api_key="tavily_key",
            enable_vector_memory=True
        )
        
        # 创建测试任务
        task = {
            'task_id': 1,
            'description': '搜索AI相关内容',
            'search_queries': ['artificial intelligence'],
            'sources': ['tavily']
        }
        state = self._create_test_state()
        
        # 执行任务
        updated_state = researcher.execute_task(state, task)
        
        # 验证搜索调用
        mock_tavily_instance.search.assert_called_with('artificial intelligence')
        
        # 验证结果存储
        mock_memory_instance.store_research_result.assert_called_once()
        
        # 验证状态更新
        assert len(updated_state['research_results']) == 1
    
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_search_tavily(self, mock_prompt_loader, mock_vector_memory):
        """测试Tavily搜索"""
        mock_tavily_search = Mock()
        mock_tavily_search.search.return_value = {'query': 'test', 'results': []}
        
        with patch('RAgents.agents.researcher.TavilySearch', return_value=mock_tavily_search):
            researcher = Researcher(llm=self.mock_llm, tavily_api_key="test_key")
            
            result = researcher._search("AI development", "tavily")
            
            mock_tavily_search.search.assert_called_once_with("AI development")
            assert result['query'] == 'test'
    
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_search_arxiv(self, mock_prompt_loader, mock_vector_memory):
        """测试ArXiv搜索"""
        mock_arxiv_search = Mock()
        mock_arxiv_search.search.return_value = {'query': 'test', 'results': []}
        
        with patch('RAgents.agents.researcher.ArxivSearch', return_value=mock_arxiv_search):
            researcher = Researcher(llm=self.mock_llm)
            
            result = researcher._search("machine learning", "arxiv")
            
            mock_arxiv_search.search.assert_called_once_with("machine learning")
            assert result['query'] == 'test'

    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_create_cached_result(self, mock_prompt_loader, mock_vector_memory):
        """测试创建缓存结果"""
        researcher = Researcher(llm=self.mock_llm)
        
        task = {
            'task_id': 1,
            'description': '测试任务'
        }
        similar_query = {
            'similarity': 0.9,
            'quality_score': 4.0,
            'results_summary': '缓存结果摘要',
            'timestamp': '2023-01-01T00:00:00'
        }
        
        results = researcher._create_cached_result(task, similar_query)
        
        assert len(results) == 1
        result = results[0]
        assert result['task_id'] == 1
        assert result['source'] == 'cached_vector_memory'
        assert result['similarity_score'] == 0.9
        assert result['cached_quality'] == 4.0
        assert result['results'] == '缓存结果摘要'
    
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_add_results_to_state(self, mock_prompt_loader, mock_vector_memory):
        """测试添加结果到状态"""
        researcher = Researcher(llm=self.mock_llm)
        
        results = [
            {'query': 'test1', 'results': ['result1']},
            {'query': 'test2', 'results': ['result2']}
        ]
        task = {
            'task_id': 1,
            'description': '测试任务'
        }
        
        # 创建包含研究计划的状态
        state = self._create_test_state()
        state['research_plan'] = {
            'sub_tasks': [
                {'task_id': 1, 'status': 'pending'},
                {'task_id': 2, 'status': 'pending'}
            ]
        }
        
        researcher._add_results_to_state(state, results, task)
        
        # 验证结果被添加
        assert len(state['research_results']) == 2
        
        # 验证任务状态被更新
        tasks = state['research_plan']['sub_tasks']
        completed_task = next(t for t in tasks if t['task_id'] == 1)
        assert completed_task['status'] == 'completed'
        
        # 验证其他任务状态不变
        other_task = next(t for t in tasks if t['task_id'] == 2)
        assert other_task['status'] == 'pending'
    
    @patch('RAgents.agents.researcher.TavilySearch')
    @patch('RAgents.agents.researcher.VectorMemory')
    @patch('RAgents.agents.researcher.PromptLoader')
    def test_execute_task_respects_request_limits(self, mock_prompt_loader, mock_vector_memory,
                                                  mock_tavily_search):
        """测试执行任务时遵守请求限制"""
        mock_tavily_instance = Mock()
        mock_tavily_instance.search.return_value = {'results': []}
        mock_tavily_search.return_value = mock_tavily_instance
        
        mock_memory_instance = Mock()
        mock_memory_instance.find_similar_queries.return_value = []  # 无缓存命中
        mock_vector_memory.return_value = mock_memory_instance
        
        researcher = Researcher(
            llm=self.mock_llm,
            tavily_api_key="test_key",
            enable_vector_memory=True
        )
        researcher.max_requests_per_task = 2  # 设置低限制进行测试
        
        # 创建包含多个查询和源的任务
        task = {
            'task_id': 1,
            'description': '多查询任务',
            'search_queries': ['query1', 'query2', 'query3'],  # 3个查询
            'sources': ['tavily', 'arxiv']  # 2个源
        }
        state = self._create_test_state()
        
        researcher.execute_task(state, task)
        
        # 应该最多执行2次搜索请求
        assert mock_tavily_instance.search.call_count <= 2
    
    def _create_test_state(self) -> ResearchState:
        """创建测试用的ResearchState"""
        return {
            'query': '人工智能发展趋势',
            'query_type': 'RESEARCH',
            'research_plan': None,
            'plan_approved': True,
            'research_results': [],
            'current_task': None,
            'iteration_count': 0,
            'max_iterations': 3,
            'needs_more_research': True,
            'final_report': None,
            'output_format': 'markdown',
            'current_step': 'researching',
            'user_feedback': None,
            'auto_approve_plan': False,
            'simple_response': None
        }