import pytest
from unittest.mock import Mock, patch
from RAgents.agents.planner import Planner
from RAgents.llms.base import BaseLLM
from RAgents.workflow.state import ResearchState


class MockLLM(BaseLLM):
    """用于测试的模拟 LLM 类"""
    
    def __init__(self, api_key: str = "test_key", model: str = "test_model", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.responses = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """返回预设的响应或默认值"""
        if self.responses:
            return self.responses.pop(0)
        return '{"research_goal": "test goal", "sub_tasks": [], "completion_criteria": "", "estimated_iterations": 1}'
    
    def stream_generate(self, prompt: str, **kwargs):
        """模拟流式生成"""
        yield "Test response"
    
    def set_responses(self, responses: list):
        """设置预设的响应列表"""
        self.responses = responses.copy()


class TestPlanner:
    """测试 Planner 类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.mock_llm = MockLLM()
        with patch('RAgents.agents.planner.PromptLoader'):
            self.planner = Planner(self.mock_llm)
    
    def test_init(self):
        """测试初始化"""
        assert self.planner.llm == self.mock_llm
        assert self.planner.prompt_loader is not None
    
    def test_create_research_plan_success(self):
        """测试成功创建研究计划"""
        # 设置模拟响应
        mock_response = '''
        {
            "research_goal": "研究人工智能发展历史",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "研究AI早期发展",
                    "search_queries": ["人工智能历史", "AI发展"],
                    "sources": ["tavily"],
                    "status": "pending",
                    "priority": 1
                }
            ],
            "completion_criteria": "收集足够的历史信息",
            "estimated_iterations": 3
        }
        '''
        self.mock_llm.set_responses([mock_response])
        
        state = self._create_test_state()
        updated_state = self.planner.create_research_plan(state)
        
        assert updated_state['research_plan'] is not None
        assert updated_state['research_plan']['research_goal'] == "研究人工智能发展历史"
        assert updated_state['research_plan']['estimated_iterations'] == 3
        assert updated_state['max_iterations'] == 3
    
    def test_create_research_plan_with_user_feedback(self):
        """测试带用户反馈的研究计划创建"""
        self.mock_llm.set_responses(['{"research_goal": "test goal", "sub_tasks": [], "completion_criteria": "", "estimated_iterations": 1}'])
        
        state = self._create_test_state()
        state['user_feedback'] = '请重点关注最近的进展'
        
        with patch.object(self.planner.prompt_loader, 'load') as mock_load:
            mock_load.return_value = 'mock prompt'
            self.planner.create_research_plan(state)
            
            # 验证用户反馈被传递
            mock_load.assert_called_with(
                'planner_create_plan',
                query=state['query'],
                user_feedback='请重点关注最近的进展'
            )
    
    def test_create_research_plan_filters_sources(self):
        """测试来源过滤功能"""
        # 包含无效来源的响应
        mock_response = '''
        {
            "research_goal": "测试目标",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "任务1",
                    "search_queries": ["查询1"],
                    "sources": ["invalid_source", "tavily"],
                    "status": "pending",
                    "priority": 1
                },
                {
                    "task_id": 2,
                    "description": "任务2",
                    "search_queries": ["查询2"],
                    "status": "pending"
                }
            ],
            "completion_criteria": "完成标准",
            "estimated_iterations": 3
        }
        '''
        self.mock_llm.set_responses([mock_response])
        
        state = self._create_test_state()
        updated_state = self.planner.create_research_plan(state)
        
        # 检查来源过滤
        tasks = updated_state['research_plan']['sub_tasks']
        assert tasks[0]['sources'] == ["tavily"]  # 无效来源被过滤
        assert tasks[1]['sources'] == ["tavily"]  # 缺失sources时使用默认值
    
    def test_create_research_plan_json_decode_error(self):
        """测试JSON解析错误时使用fallback计划"""
        self.mock_llm.set_responses(['invalid json response'])
        
        state = self._create_test_state()
        updated_state = self.planner.create_research_plan(state)
        
        # 应该使用fallback计划
        assert updated_state['research_plan'] is not None
        assert updated_state['research_plan']['research_goal'] == state['query']
        assert len(updated_state['research_plan']['sub_tasks']) == 1
        assert updated_state['research_plan']['sub_tasks'][0]['task_id'] == 1
    
    def test_create_fallback_plan(self):
        """测试fallback计划创建"""
        query = "人工智能发展"
        fallback_plan = self.planner._create_fallback_plan(query)
        
        assert fallback_plan['research_goal'] == query
        assert len(fallback_plan['sub_tasks']) == 1
        assert fallback_plan['sub_tasks'][0]['description'] == f'Research: {query}'
        assert fallback_plan['sub_tasks'][0]['sources'] == ['tavily']
        assert fallback_plan['sub_tasks'][0]['status'] == 'pending'
        assert fallback_plan['completion_criteria'] == 'Gather sufficient information to answer the query'
        assert fallback_plan['estimated_iterations'] == 2
    
    def test_modify_plan_success(self):
        """测试成功修改计划"""
        current_plan = {
            "research_goal": "原目标",
            "sub_tasks": [],
            "completion_criteria": "原标准",
            "estimated_iterations": 2
        }
        
        mock_modified_plan = {
            "research_goal": "修改后目标",
            "sub_tasks": [],
            "completion_criteria": "新标准", 
            "estimated_iterations": 3
        }
        
        self.mock_llm.set_responses([str(mock_modified_plan).replace("'", '"')])
        
        state = self._create_test_state()
        state['research_plan'] = current_plan
        
        updated_state = self.planner.modify_plan(state, "请增加更多细节")
        
        assert updated_state['research_plan']['research_goal'] == "修改后目标"
        assert updated_state['research_plan']['estimated_iterations'] == 3
    
    def test_modify_plan_json_decode_error(self):
        """测试修改计划时JSON解析错误"""
        current_plan = {
            "research_goal": "原目标",
            "sub_tasks": [],
            "completion_criteria": "原标准",
            "estimated_iterations": 2
        }
        
        self.mock_llm.set_responses(['invalid json'])
        
        state = self._create_test_state()
        state['research_plan'] = current_plan
        
        updated_state = self.planner.modify_plan(state, "修改请求")
        
        # 计划应该保持不变
        assert updated_state['research_plan'] == current_plan
    
    def test_evaluate_context_sufficiency_max_iterations_reached(self):
        """测试达到最大迭代次数时的上下文充分性评估"""
        plan = {
            "research_goal": "测试目标",
            "sub_tasks": [],
            "completion_criteria": "标准",
            "estimated_iterations": 3
        }
        
        state = self._create_test_state()
        state['research_plan'] = plan
        state['iteration_count'] = 3  # 达到最大迭代次数
        state['max_iterations'] = 3
        state['research_results'] = []  # 没有结果也返回True
        
        assert self.planner.evaluate_context_sufficiency(state) is True
    
    def test_evaluate_context_sufficiency_no_results(self):
        """测试没有结果时的上下文充分性评估"""
        plan = {
            "research_goal": "测试目标",
            "sub_tasks": [],
            "completion_criteria": "标准",
            "estimated_iterations": 3
        }
        
        state = self._create_test_state()
        state['research_plan'] = plan
        state['iteration_count'] = 1
        state['max_iterations'] = 3
        state['research_results'] = []
        
        assert self.planner.evaluate_context_sufficiency(state) is False
    
    def test_evaluate_context_sufficiency_auto_sufficient(self):
        """测试自动判定充分的条件"""
        plan = {
            "research_goal": "测试目标",
            "sub_tasks": [],
            "completion_criteria": "标准",
            "estimated_iterations": 5
        }
        
        state = self._create_test_state()
        state['research_plan'] = plan
        state['iteration_count'] = 2
        state['max_iterations'] = 5
        state['research_results'] = [{}] * 12  # 12个结果，超过10个
        
        assert self.planner.evaluate_context_sufficiency(state) is True
    
    def test_evaluate_context_sufficiency_llm_decision(self):
        """测试需要LLM决策的上下文充分性评估"""
        plan = {
            "research_goal": "测试目标",
            "sub_tasks": [],
            "completion_criteria": "标准",
            "estimated_iterations": 5
        }
        
        self.mock_llm.set_responses(['YES'])  # LLM返回YES
        
        state = self._create_test_state()
        state['research_plan'] = plan
        state['iteration_count'] = 1
        state['max_iterations'] = 5
        state['research_results'] = [{}] * 5  # 5个结果，不满足自动充分条件
        
        result = self.planner.evaluate_context_sufficiency(state)
        assert result is True
    
    def test_get_next_task_available(self):
        """测试获取下一个可用任务"""
        plan = {
            "sub_tasks": [
                {"task_id": 3, "description": "任务3", "status": "pending", "priority": 1},
                {"task_id": 1, "description": "任务1", "status": "completed", "priority": 2},
                {"task_id": 2, "description": "任务2", "status": "pending", "priority": 3}
            ]
        }
        
        state = self._create_test_state()
        state['research_plan'] = plan
        
        next_task = self.planner.get_next_task(state)
        
        # 应该返回优先级最高的pending任务（task_id=3, priority=1）
        assert next_task is not None
        assert next_task['task_id'] == 3
        assert next_task['status'] == 'pending'
    
    def test_get_next_task_no_pending(self):
        """测试没有pending任务时返回None"""
        plan = {
            "sub_tasks": [
                {"task_id": 1, "description": "任务1", "status": "completed", "priority": 1},
                {"task_id": 2, "description": "任务2", "status": "completed", "priority": 2}
            ]
        }
        
        state = self._create_test_state()
        state['research_plan'] = plan
        
        next_task = self.planner.get_next_task(state)
        
        assert next_task is None
    
    def test_get_next_task_no_plan(self):
        """测试没有计划时返回None"""
        state = self._create_test_state()
        state['research_plan'] = None
        
        next_task = self.planner.get_next_task(state)
        
        assert next_task is None
    
    def test_format_plan_for_display(self):
        """测试计划显示格式化"""
        plan = {
            "research_goal": "研究AI发展",
            "completion_criteria": "收集足够信息",
            "estimated_iterations": 3,
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "研究历史",
                    "search_queries": ["AI历史", "人工智能发展"],
                    "sources": ["tavily"],
                    "priority": 1,
                    "status": "pending"
                }
            ]
        }
        
        display_text = self.planner.format_plan_for_display(plan)
        
        assert "研究目标: 研究AI发展" in display_text
        assert "预计迭代次数: 3" in display_text
        assert "完成标准: 收集足够信息" in display_text
        assert "子任务列表:" in display_text
        assert "1. 研究历史" in display_text
        assert "Queries: AI历史, 人工智能发展" in display_text
        assert "Sources: tavily" in display_text
        assert "Priority: 1" in display_text
        assert "Status: pending" in display_text
    
    def test_format_plan_for_display_missing_fields(self):
        """测试格式化显示时缺少字段的默认值处理"""
        plan = {
            "research_goal": "测试目标",
            "sub_tasks": [
                {
                    "task_id": 1,
                    "description": "测试任务"
                    # 缺少其他字段
                }
            ]
            # 缺少completion_criteria, estimated_iterations
        }
        
        display_text = self.planner.format_plan_for_display(plan)
        
        assert "研究目标: 测试目标" in display_text
        assert "预计迭代次数: N/A" in display_text
        assert "完成标准: N/A" in display_text
        assert "Queries: " in display_text  # 空列表
        assert "Sources: " in display_text   # 空列表
        assert "Priority: N/A" in display_text
        assert "Status: pending" in display_text  # 默认值
    
    def _create_test_state(self) -> ResearchState:
        """创建测试用的ResearchState"""
        return {
            'query': '人工智能发展历史',
            'query_type': 'RESEARCH',
            'research_plan': None,
            'plan_approved': False,
            'research_results': [],
            'current_task': None,
            'iteration_count': 0,
            'max_iterations': 3,
            'needs_more_research': True,
            'final_report': None,
            'output_format': 'markdown',
            'current_step': 'planning',
            'user_feedback': None,
            'auto_approve_plan': False,
            'simple_response': None
        }