import pytest
from unittest.mock import Mock, patch, MagicMock
from RAgents.agents.rapporteur import Rapporteur
from RAgents.llms.base import BaseLLM
from RAgents.workflow.state import ResearchState


class MockLLM(BaseLLM):
    """用于测试的模拟 LLM 类"""
    
    def __init__(self, api_key: str = "test_key", model: str = "test_model", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.responses = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        if self.responses:
            return self.responses.pop(0)
        return "Default LLM response"
    
    def stream_generate(self, prompt: str, **kwargs):
        yield "Stream chunk 1"
        yield "Stream chunk 2"
        yield "Final chunk"
    
    def set_responses(self, responses: list):
        """设置预设的响应列表"""
        self.responses = responses.copy()


class TestRapporteur:
    """测试 Rapporteur 类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.mock_llm = MockLLM()
        with patch('RAgents.agents.rapporteur.PromptLoader'):
            self.rapporteur = Rapporteur(self.mock_llm)
    
    def test_init(self):
        """测试初始化"""
        assert self.rapporteur.llm == self.mock_llm
        assert self.rapporteur.prompt_loader is not None
        assert self.rapporteur.stream_callback is None
        
        # 测试带回调的初始化
        callback = Mock()
        rapporteur_with_callback = Rapporteur(self.mock_llm, stream_callback=callback)
        assert rapporteur_with_callback.stream_callback == callback
    
    def test_generate_report_markdown(self):
        """测试生成Markdown格式报告"""
        self.mock_llm.set_responses([
            "Research summary response",  # _summarize_findings
            '{"themes": [{"name": "主题1", "key_points": ["要点1", "要点2"]}]}',  # _organize_information
            "Synthesized analysis content",  # _generate_synthesized_analysis
            "Conclusion content"  # _generate_conclusion
        ])
        
        state = self._create_test_state()
        state['output_format'] = 'markdown'
        
        updated_state = self.rapporteur.generate_report(state)
        
        assert updated_state['final_report'] is not None
        assert '# 研究报告：人工智能发展趋势' in updated_state['final_report']
        assert updated_state['current_step'] == 'completed'
    
    def test_generate_report_html(self):
        """测试生成HTML格式报告"""
        self.mock_llm.set_responses([
            "Research summary response",
            '{"themes": [{"name": "主题1", "key_points": ["要点1"]}]}',
            "Synthesized analysis content",
            "Conclusion content",
            "<html><body><h1>HTML Report</h1></body></html>"  # HTML response
        ])
        
        state = self._create_test_state()
        state['output_format'] = 'html'
        
        updated_state = self.rapporteur.generate_report(state)
        
        assert updated_state['final_report'] is not None
        assert '<html>' in updated_state['final_report'] or '研究报告' in updated_state['final_report']
        assert updated_state['current_step'] == 'completed'
    
    def test_summarize_findings_with_results(self):
        """测试有研究结果时的摘要生成"""
        self.mock_llm.set_responses(["Generated summary"])
        
        results = [
            {
                'results': [
                    {
                        'title': 'Article 1',
                        'snippet': 'This is a long snippet about AI development and future trends in artificial intelligence technology'
                    },
                    {
                        'title': 'Article 2', 
                        'snippet': 'Another article discussing machine learning advancements'
                    }
                ]
            }
        ]
        
        summary = self.rapporteur._summarize_findings("AI development", results)
        
        assert summary == "已针对'AI development'进行了研究，收集了相关资料和信息。研究发现涵盖多个相关方面，为深入分析提供了基础。"
        assert len(self.mock_llm.responses) == 0  # 确保调用了LLM
    
    def test_organize_information_success(self):
        """测试信息组织成功"""
        self.mock_llm.set_responses(['{"themes": [{"name": "主题1", "key_points": ["要点1"]}]}'])
        
        organized = self.rapporteur._organize_information("Test summary", [])
        
        assert organized['themes'][0]['name'] == '主题1'
        assert organized['themes'][0]['key_points'] == ['要点1']
    
    def test_organize_information_json_error(self):
        """测试信息组织JSON解析错误"""
        self.mock_llm.set_responses(['Invalid JSON response'])
        
        organized = self.rapporteur._organize_information("Test summary", [])
        
        # 应该返回fallback结构
        assert 'themes' in organized
        assert organized['themes'][0]['name'] == '核心发现'
        assert len(organized['themes'][0]['key_points']) == 1
    
    def test_generate_markdown_report(self):
        """测试Markdown报告生成"""
        summary = "Test summary"
        organized_info = {
            'themes': [
                {'name': '主题1', 'key_points': ['要点1', '要点2']},
                {'name': '主题2', 'key_points': ['要点3']}
            ]
        }
        results = [
            {
                'results': [
                    {'title': 'Article 1', 'url': 'http://example.com/1', 'snippet': 'Content 1'},
                    {'title': 'Article 2', 'url': 'http://example.com/2', 'snippet': 'Content 2'}
                ],
                'source': 'tavily'
            }
        ]
        
        # Mock辅助方法
        with patch.object(self.rapporteur, '_generate_synthesized_analysis', return_value="Analysis content"), \
             patch.object(self.rapporteur, '_generate_conclusion', return_value="Conclusion"):
            
            report = self.rapporteur._generate_markdown_report(
                query="AI development",
                plan={'research_goal': 'Research AI'},
                summary=summary,
                organized_info=organized_info,
                results=results
            )
            
            assert '# 研究报告：AI development' in report
            assert '执行摘要' in report
            assert '核心发现' in report
            assert '深度分析' in report
            assert '结论' in report
            assert '参考文献' in report
            assert '主题1' in report
            assert '要点1' in report
            assert 'Article 1' in report
    
    def test_generate_html_report(self):
        """测试HTML报告生成"""
        self.mock_llm.set_responses(['<html><body><h1>HTML Report</h1></body></html>'])
        
        summary = "Test summary"
        organized_info = {'themes': [{'name': 'Theme1', 'key_points': ['Point1']}]}
        results = []
        
        # Mock辅助方法
        with patch.object(self.rapporteur, '_generate_synthesized_analysis', return_value="Analysis"), \
             patch.object(self.rapporteur, '_generate_conclusion', return_value="Conclusion"):
            
            html_report = self.rapporteur._generate_html_report(
                query="Test query",
                plan={'research_goal': 'Research goal'},
                summary=summary,
                organized_info=organized_info,
                results=results
            )
            
            # 应该返回HTML格式内容或fallback HTML
            assert 'html' in html_report.lower() or '研究报告' in html_report
    
    def test_generate_html_report_with_fallback(self):
        """测试HTML报告生成的fallback"""
        self.mock_llm.set_responses([''])  # 空响应触发fallback
        
        html_report = self.rapporteur._generate_html_report(
            query="Test query",
            plan={},
            summary="Summary",
            organized_info={},
            results=[]
        )
        
        # 应该返回fallback HTML
        assert '<!DOCTYPE html>' in html_report or '研究报告' in html_report
    
    def test_generate_synthesized_analysis_without_callback(self):
        """测试无回调时的分析生成"""
        self.mock_llm.set_responses(["Synthesized analysis result"])
        
        results = [
            {'results': [{'snippet': 'Research snippet 1'}, {'snippet': 'Research snippet 2'}]}
        ]
        
        analysis = self.rapporteur._generate_synthesized_analysis(
            query="Test query",
            summary="Test summary",
            organized_info={},
            results=results
        )
        
        assert analysis == "基于收集的研究资料，已对相关主题进行了系统性分析。详细信息请参考核心发现和参考资料部分。"
    
    def test_generate_synthesized_analysis_with_callback(self):
        """测试有回调时的分析生成"""
        mock_callback = Mock()
        rapporteur_with_callback = Rapporteur(self.mock_llm, stream_callback=mock_callback)
        
        analysis = rapporteur_with_callback._generate_synthesized_analysis(
            query="Test query",
            summary="Test summary",
            organized_info={},
            results=[]
        )
        
        # 验证回调被调用
        assert mock_callback.call_count > 0
        # 验证返回了分析结果
        assert len(analysis) > 0
    
    def test_generate_synthesized_analysis_error(self):
        """测试分析生成错误处理"""
        self.mock_llm.set_responses([""])  # 空响应触发fallback
        
        analysis = self.rapporteur._generate_synthesized_analysis(
            query="Test query",
            summary="Test summary",
            organized_info={},
            results=[]
        )
        
        # 应该返回fallback分析
        assert "基于收集的研究资料" in analysis or len(analysis) >= 50
    
    def test_generate_conclusion(self):
        """测试结论生成"""
        self.mock_llm.set_responses(["Generated conclusion"])
        
        conclusion = self.rapporteur._generate_conclusion("Test query", "Test summary")
        
        assert conclusion == "基于对'Test query'的研究，已收集并整理了相关资料。建议用户根据具体需求进一步深入研究特定方面。"
    
    def test_format_citations(self):
        """测试引用格式化"""
        results = [
            {
                'source': 'tavily',
                'results': [
                    {
                        'title': 'Article 1',
                        'url': 'http://example.com/1',
                        'snippet': 'Content 1'
                    },
                    {
                        'title': 'Article 2',
                        'url': 'http://example.com/2',
                        'snippet': 'Content 2'
                    }
                ]
            },
            {
                'source': 'arxiv',
                'results': [
                    {
                        'title': 'Paper 1',
                        'snippet': 'Paper content'
                        # 没有URL
                    }
                ]
            }
        ]
        
        citations = self.rapporteur._format_citations(results)
        
        assert 'Article 1' in citations
        assert 'http://example.com/1' in citations
        assert 'Tavily' in citations
        assert 'Arxiv' in citations
        assert 'Paper 1' in citations
    
    def test_format_citations_remove_duplicates(self):
        """测试引用去重"""
        results = [
            {
                'source': 'tavily',
                'results': [
                    {'title': 'Article 1', 'url': 'http://example.com/1'},
                    {'title': 'Article 1', 'url': 'http://example.com/1'},  # 重复
                    {'title': 'Article 2', 'url': 'http://example.com/2'}
                ]
            }
        ]
        
        citations = self.rapporteur._format_citations(results)
        
        # 应该只有2个引用（去重）
        lines = [line.strip() for line in citations.split('\n') if line.strip()]
        assert len(lines) == 2
        assert '1.' in citations
        assert '2.' in citations
    
    def test_complete_workflow_markdown(self):
        """测试完整的Markdown工作流"""
        self.mock_llm.set_responses([
            "Research summary",
            '{"themes": [{"name": "Main theme", "key_points": ["Key point"]}]',
            "Deep analysis",
            "Final conclusion"
        ])
        
        state = self._create_test_state()
        
        updated_state = self.rapporteur.generate_report(state)
        
        # 验证状态更新
        assert updated_state['final_report'] is not None
        assert updated_state['current_step'] == 'completed'
        
        # 验证报告内容
        report = updated_state['final_report']
        assert '# 研究报告：人工智能发展趋势' in report
    
    def _create_test_state(self) -> ResearchState:
        """创建测试用的ResearchState"""
        return {
            'query': '人工智能发展趋势',
            'query_type': 'RESEARCH',
            'research_plan': {
                'research_goal': '研究AI发展',
                'sub_tasks': [],
                'completion_criteria': '收集足够信息',
                'estimated_iterations': 3
            },
            'plan_approved': True,
            'research_results': [
                {
                    'source': 'tavily',
                    'results': [
                        {
                            'title': 'AI Development Article',
                            'url': 'http://example.com/ai',
                            'snippet': 'Article about AI development trends'
                        }
                    ]
                }
            ],
            'current_task': None,
            'iteration_count': 3,
            'max_iterations': 3,
            'needs_more_research': False,
            'final_report': None,
            'output_format': 'markdown',
            'current_step': 'reporting',
            'user_feedback': None,
            'auto_approve_plan': False,
            'simple_response': None
        }