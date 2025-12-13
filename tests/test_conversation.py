import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from RAgents.agents.conversation import ConversationManager


class MockLLM:
    """用于测试的模拟 LLM 类"""
    
    def __init__(self, api_key: str = "test_key", model: str = "test_model", **kwargs):
        self.api_key = api_key
        self.model = model
        self.responses = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """返回预设的响应或默认值"""
        if self.responses:
            return self.responses.pop(0)
        return "这是一个测试响应"
    
    def set_responses(self, responses: list):
        """设置预设的响应列表"""
        self.responses = responses.copy()


class MockTavilySearch:
    """模拟Tavily搜索工具"""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def search(self, query, max_results=None):
        return {
            'results': [
                {
                    'title': f'测试搜索结果1: {query}',
                    'snippet': f'关于{query}的测试摘要1',
                    'url': 'https://example.com/result1'
                },
                {
                    'title': f'测试搜索结果2: {query}',
                    'snippet': f'关于{query}的测试摘要2',
                    'url': 'https://example.com/result2'
                }
            ]
        }


class MockArxivSearch:
    """模拟Arxiv搜索工具"""
    
    def search(self, query, max_results=None):
        return {
            'results': [
                {
                    'title': f'测试论文1: {query}',
                    'snippet': f'关于{query}的论文摘要1',
                    'url': 'https://arxiv.org/example1'
                }
            ]
        }


class MockVectorMemory:
    """模拟向量记忆工具"""
    
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory
        self.similar_queries = []
        self.stored_results = []
    
    def find_similar_queries(self, query, threshold=0.8, limit=3):
        return self.similar_queries
    
    def store_research_result(self, query, results, quality_score=0.0, metadata=None):
        self.stored_results.append({
            'query': query,
            'results': results,
            'quality_score': quality_score,
            'metadata': metadata
        })
    
    def set_similar_queries(self, queries):
        self.similar_queries = queries


@pytest.fixture
def conversation_manager():
    """创建ConversationManager测试实例"""
    config = {
        'llm_provider': 'test_provider',
        'llm_api_key': 'test_api_key',
        'llm_model': 'test_model',
        'tavily_api_key': 'test_tavily_key',
        'mcp_server_url': None,
        'mcp_api_key': None,
        'vector_memory_path': './test_vector_memory'
    }
    
    with patch('RAgents.agents.conversation.LLMFactory') as mock_llm_factory, \
         patch('RAgents.agents.conversation.TavilySearch', MockTavilySearch), \
         patch('RAgents.agents.conversation.ArxivSearch', MockArxivSearch), \
         patch('RAgents.agents.conversation.VectorMemory', MockVectorMemory), \
         patch('RAgents.agents.conversation.PromptLoader'):
        
        mock_llm = MockLLM()
        mock_llm_factory.create_llm.return_value = mock_llm
        
        manager = ConversationManager(config)
        return manager

class TestConversationManager:
    """测试ConversationManager类的主要功能"""
    
    def test_init(self, conversation_manager):
        """测试初始化"""
        assert conversation_manager.config['llm_provider'] == 'test_provider'
        assert conversation_manager.context_window == 5
        assert conversation_manager.relevance_threshold == 0.8
        assert len(conversation_manager.conversation_history) == 0
    
    def test_analyze_intent_simple_search(self, conversation_manager):
        """测试简单搜索意图分析"""
        # 测试包含搜索关键词的查询
        assert conversation_manager._analyze_intent("搜索最新AI技术") == 'simple_search'
        assert conversation_manager._analyze_intent("search Python教程") == 'simple_search'
        assert conversation_manager._analyze_intent("查找机器学习资料") == 'simple_search'
    
    def test_analyze_intent_complex_research(self, conversation_manager):
        """测试复杂研究意图分析"""
        # 测试包含研究指标词的查询
        assert conversation_manager._analyze_intent("分析最新AI技术") == 'complex_research'
        assert conversation_manager._analyze_intent("详细研究机器学习算法") == 'complex_research'
        assert conversation_manager._analyze_intent("全面分析深度学习框架") == 'complex_research'
        
        # 测试同时包含搜索和研究关键词的查询
        assert conversation_manager._analyze_intent("搜索并详细分析深度学习") == 'complex_research'
    
    def test_analyze_intent_conversation(self, conversation_manager):
        """测试对话意图分析"""
        # 测试普通对话查询
        assert conversation_manager._analyze_intent("你好") == 'conversation'
        assert conversation_manager._analyze_intent("天气怎么样") == 'conversation'
        assert conversation_manager._analyze_intent("什么是机器学习") == 'conversation'
    
    def test_extract_search_query(self, conversation_manager):
        """测试搜索查询提取"""
        # 测试提取搜索查询
        assert conversation_manager._extract_search_query("搜索Python教程") == "Python教程"
        assert conversation_manager._extract_search_query("查找关于机器学习的资料") == "机器学习的资料"
        assert conversation_manager._extract_search_query("find最新AI技术") == "最新AI技术"
        assert conversation_manager._extract_search_query("最新AI技术") == "最新AI技术"
    
    def test_handle_direct_search_with_tavily(self, conversation_manager):
        """测试使用Tavily的直接搜索"""
        response = conversation_manager._handle_direct_search("Python教程")
        
        assert "测试搜索结果1: Python教程" in response
        assert "测试搜索结果2: Python教程" in response
        assert "https://example.com/result1" in response
    
    @patch('RAgents.agents.conversation.TavilySearch', None)
    def test_handle_direct_search_with_arxiv(self, conversation_manager):
        """测试使用Arxiv的直接搜索（当Tavily不可用时）"""
        response = conversation_manager._handle_direct_search("机器学习算法")

        assert "1. 测试搜索结果" in response
        assert "https://example.com/result1" in response
    
    def test_handle_conversation_query(self, conversation_manager):
        """测试对话查询处理"""
        # 设置模拟的相似查询
        conversation_manager.vector_memory.set_similar_queries([
            {
                'query': '机器学习基础',
                'results_summary': '机器学习是AI的一个分支',
                'similarity': 0.9,
                'quality_score': 4.5
            }
        ])
        
        # 设置LLM响应
        conversation_manager.llm.set_responses(["这是一个关于机器学习的基础知识"])
        
        # 添加一些对话历史
        conversation_manager.conversation_history = [
            {
                'role': 'user',
                'content': '你好',
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'assistant',
                'content': '你好！有什么可以帮助你的吗？',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        response = conversation_manager._handle_conversation_query("什么是机器学习")
        
        assert "这是一个关于机器学习的基础知识" == response
    
    def test_process_user_input(self, conversation_manager):
        """测试用户输入处理"""
        # 设置模拟响应
        conversation_manager.llm.set_responses(["这是一个关于AI的介绍"])
        
        # 处理简单搜索请求
        conversation_manager._process_user_input("搜索AI技术")
        
        # 验证对话历史
        assert len(conversation_manager.conversation_history) == 2
        assert conversation_manager.conversation_history[0]['role'] == 'user'
        assert conversation_manager.conversation_history[0]['content'] == '搜索AI技术'
        assert conversation_manager.conversation_history[1]['role'] == 'assistant'
    
    def test_is_exit_command(self, conversation_manager):
        """测试退出命令识别"""
        # 测试各种退出命令
        assert conversation_manager._is_exit_command("exit") == True
        assert conversation_manager._is_exit_command("quit") == True
        assert conversation_manager._is_exit_command("退出") == True
        assert conversation_manager._is_exit_command("结束") == True
        assert conversation_manager._is_exit_command("bye") == True
        assert conversation_manager._is_exit_command("goodbye") == True
        assert conversation_manager._is_exit_command("EXIT") == True  # 测试大小写
        
        # 测试非退出命令
        assert conversation_manager._is_exit_command("搜索AI") == False
        assert conversation_manager._is_exit_command("你好") == False
        assert conversation_manager._is_exit_command("") == False
    
    def test_get_conversation_context(self, conversation_manager):
        """测试对话上下文获取"""
        # 测试空历史
        assert conversation_manager._get_conversation_context() == ""
        
        # 添加一些对话历史
        conversation_manager.conversation_history = [
            {
                'role': 'user',
                'content': '你好',
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'assistant',
                'content': '你好！有什么可以帮助你的吗？',
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'user',
                'content': '什么是机器学习',
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'assistant',
                'content': '机器学习是AI的一个分支',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        context = conversation_manager._get_conversation_context()
        assert "用户: 你好" in context
        assert "系统: 你好！有什么可以帮助你的吗？" in context
        assert "用户: 什么是机器学习" in context
        assert "系统: 机器学习是AI的一个分支" in context