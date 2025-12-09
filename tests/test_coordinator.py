import pytest
from unittest.mock import Mock, patch, MagicMock
from RAgents.agents.coordinator import Coordinator
from RAgents.llms.base import BaseLLM


class MockLLM(BaseLLM):
    """用于测试的模拟 LLM 类"""
    
    def __init__(self, api_key: str = "test_key", model: str = "test_model", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.responses = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """返回预设的响应或默认值"""
        if self.responses:
            return self.responses.pop(0)
        return "RESEARCH"
    
    def stream_generate(self, prompt: str, **kwargs):
        """模拟流式生成"""
        yield "Test response"
    
    def set_responses(self, responses: list):
        """设置预设的响应列表"""
        self.responses = responses.copy()


class TestCoordinator:
    """测试 Coordinator 类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.mock_llm = MockLLM()
        with patch('RAgents.agents.coordinator.PromptLoader'):
            self.coordinator = Coordinator(self.mock_llm)
    
    def test_init(self):
        """测试初始化"""
        assert self.coordinator.llm == self.mock_llm
        assert self.coordinator.prompt_loader is not None
    
    def test_initialize_research_basic(self):
        """测试基本的研究初始化"""
        state = self.coordinator.initialize_research("test query")
        
        assert state['query'] == "test query"
        assert state['query_type'] == "RESEARCH"
        assert state['research_plan'] is None
        assert state['plan_approved'] is False
        assert state['research_results'] == []
        assert state['current_task'] is None
        assert state['iteration_count'] == 0
        assert state['max_iterations'] == 5
        assert state['needs_more_research'] is True
        assert state['final_report'] is None
        assert state['output_format'] == "markdown"
        assert state['current_step'] == 'initializing'
        assert state['user_feedback'] is None
        assert state['auto_approve_plan'] is False
        assert state['simple_response'] is None
    
    def test_initialize_research_with_options(self):
        """测试使用自定义选项初始化研究"""
        state = self.coordinator.initialize_research(
            "test query", 
            auto_approve=True, 
            output_format="pdf"
        )
        
        assert state['auto_approve_plan'] is True
        assert state['output_format'] == "pdf"
    
    def test_initialize_research_greeting(self):
        """测试问候类型的查询"""
        self.mock_llm.set_responses(["GREETING"])
        
        state = self.coordinator.initialize_research("你好")
        
        assert state['query_type'] == "GREETING"
        assert state['current_step'] == 'completed'
        assert state['needs_more_research'] is False
        assert state['simple_response'] is not None
    
    def test_initialize_research_inappropriate(self):
        """测试不适当类型的查询"""
        self.mock_llm.set_responses(["INAPPROPRIATE"])
        
        state = self.coordinator.initialize_research("不适当的内容")
        
        assert state['query_type'] == "INAPPROPRIATE"
        assert state['current_step'] == 'completed'
        assert state['needs_more_research'] is False
        assert state['simple_response'] is not None
    
    def test_classify_query_research(self):
        """测试查询分类 - 研究类型"""
        self.mock_llm.set_responses(["RESEARCH"])
        
        query_type = self.coordinator._classify_query("人工智能的发展历史")
        
        assert query_type == "RESEARCH"
    
    def test_classify_query_greeting(self):
        """测试查询分类 - 问候类型"""
        self.mock_llm.set_responses(["GREETING"])
        
        query_type = self.coordinator._classify_query("你好")
        
        assert query_type == "GREETING"
    
    def test_classify_query_invalid_response(self):
        """测试查询分类 - 无效响应时的默认处理"""
        self.mock_llm.set_responses(["INVALID_TYPE"])
        
        query_type = self.coordinator._classify_query("some query")
        
        # 无效响应应该默认为 RESEARCH
        assert query_type == "RESEARCH"
    
    def test_classify_query_case_insensitive(self):
        """测试查询分类大小写不敏感"""
        self.mock_llm.set_responses(["greeting"])
        
        query_type = self.coordinator._classify_query("hello")
        
        assert query_type == "GREETING"  # 应该被转换为大写
    
    def test_handle_simple_query_greeting(self):
        """测试处理简单查询 - 问候"""
        self.mock_llm.set_responses(["你好！很高兴为您服务！"])
        
        response = self.coordinator._handle_simple_query("你好", "GREETING")
        
        assert response == "你好！很高兴为您服务！"
    
    def test_handle_simple_query_inappropriate(self):
        """测试处理简单查询 - 不适当内容"""
        self.mock_llm.set_responses(["我无法处理此类请求。"])
        
        response = self.coordinator._handle_simple_query("不适当内容", "INAPPROPRIATE")
        
        assert response == "我无法处理此类请求。"
    
    def test_delegate_to_planner(self):
        """测试委托给计划者"""
        initial_state = {
            'current_step': 'initializing',
            'other_field': 'test'
        }
        
        updated_state = self.coordinator.delegate_to_planner(initial_state)
        
        assert updated_state['current_step'] == 'planning'
        assert updated_state['other_field'] == 'test'  # 其他字段应该保持不变
    
    def test_delegate_to_planner_preserves_all_fields(self):
        """测试委托给计划者时保留所有字段"""
        initial_state = {
            'query': 'test query',
            'query_type': 'RESEARCH',
            'current_step': 'initializing',
            'research_results': ['result1'],
            'iteration_count': 1
        }
        
        updated_state = self.coordinator.delegate_to_planner(initial_state)
        
        # 只更新 current_step，其他字段保持不变
        assert updated_state['current_step'] == 'planning'
        assert updated_state['query'] == 'test query'
        assert updated_state['query_type'] == 'RESEARCH'
        assert updated_state['research_results'] == ['result1']
        assert updated_state['iteration_count'] == 1
    
    def test_repr(self):
        """测试字符串表示"""
        repr_str = repr(self.coordinator)
        assert "Coordinator" in repr_str
        assert "MockLLM" in repr_str
    
    @patch('RAgents.agents.coordinator.PromptLoader')
    def test_prompt_loader_initialization(self, mock_prompt_loader):
        """测试 PromptLoader 的初始化"""
        mock_loader_instance = Mock()
        mock_prompt_loader.return_value = mock_loader_instance
        
        coordinator = Coordinator(self.mock_llm)
        
        assert coordinator.prompt_loader == mock_loader_instance
        mock_prompt_loader.assert_called_once()
    
    def test_initialize_research_prompt_loader_usage(self):
        """测试初始化研究时 PromptLoader 的使用"""
        with patch.object(self.coordinator.prompt_loader, 'load') as mock_load:
            mock_load.return_value = "test prompt"
            self.mock_llm.set_responses(["RESEARCH"])
            
            self.coordinator._classify_query("test query")
            
            mock_load.assert_called_with(
                'coordinator_classify_query',
                user_query="test query"
            )
    
    def test_handle_simple_query_prompt_loader_usage(self):
        """测试处理简单查询时 PromptLoader 的使用"""
        with patch.object(self.coordinator.prompt_loader, 'load') as mock_load:
            mock_load.return_value = "test prompt"
            self.mock_llm.set_responses(["Test response"])
            
            self.coordinator._handle_simple_query("hello", "GREETING")
            
            mock_load.assert_called_with(
                'coordinator_simple_response',
                user_query="hello",
                query_type="GREETING"
            )
    
    def test_complete_research_flow(self):
        """测试完整的研究流程"""
        # 1. 初始化研究
        self.mock_llm.set_responses(["RESEARCH"])
        state = self.coordinator.initialize_research("人工智能发展趋势")
        
        assert state['current_step'] == 'initializing'
        assert state['needs_more_research'] is True
        
        # 2. 委托给计划者
        state = self.coordinator.delegate_to_planner(state)
        assert state['current_step'] == 'planning'
    
    def test_edge_case_empty_query(self):
        """测试边界情况 - 空查询"""
        self.mock_llm.set_responses(["RESEARCH"])
        
        state = self.coordinator.initialize_research("")
        
        assert state['query'] == ""
        assert state['query_type'] == "RESEARCH"
    
    def test_edge_case_very_long_query(self):
        """测试边界情况 - 非常长的查询"""
        long_query = "test" * 1000
        self.mock_llm.set_responses(["RESEARCH"])
        
        state = self.coordinator.initialize_research(long_query)
        
        assert state['query'] == long_query
        assert state['query_type'] == "RESEARCH"
    
    def test_multiple_classifications(self):
        """测试多次分类调用"""
        self.mock_llm.set_responses(["RESEARCH", "GREETING", "INAPPROPRIATE"])
        
        # 第一次调用
        type1 = self.coordinator._classify_query("query1")
        assert type1 == "RESEARCH"
        
        # 第二次调用
        type2 = self.coordinator._classify_query("query2")
        assert type2 == "GREETING"
        
        # 第三次调用
        type3 = self.coordinator._classify_query("query3")
        assert type3 == "INAPPROPRIATE"


class TestCoordinatorIntegration:
    """Coordinator 集成测试"""
    
    def test_integration_with_real_llm_responses(self):
        """测试与真实LLM响应的集成"""
        mock_llm = MockLLM()
        
        with patch('RAgents.agents.coordinator.PromptLoader') as mock_loader:
            # 设置模拟的提示加载器
            mock_loader_instance = Mock()
            mock_loader.return_value = mock_loader_instance
            mock_loader_instance.load.return_value = "mock prompt"
            
            coordinator = Coordinator(mock_llm)
            mock_llm.set_responses(["RESEARCH"])
            
            state = coordinator.initialize_research("test integration")
            
            assert state['query_type'] == "RESEARCH"
            assert state['current_step'] == 'initializing'
            mock_loader_instance.load.assert_called()