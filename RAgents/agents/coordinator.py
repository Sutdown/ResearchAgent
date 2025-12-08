from typing import Dict, Any

from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader


class Coordinator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_loader = PromptLoader()

    def initialize_research(self, user_query: str, auto_approve: bool = False, output_format: str = "markdown") -> Dict[str, Any]:
        query_type = self._classify_query(user_query)
        state = {
            'query': user_query,
            'query_type': query_type,
            # planning
            'research_plan': None,
            'plan_approved': False,
            # execute
            'research_results': [],
            'current_task': None,
            'iteration_count': 0,
            'max_iterations': 5,
            'needs_more_research': True,
            #report
            'final_report': None,
            'output_format': output_format,
            # control
            'current_step': 'initializing',
            'user_feedback': None,
            'auto_approve_plan': auto_approve,
            'simple_response': None
        }

        if query_type in ['GREETING', 'INAPPROPRIATE']:
            state['simple_response'] = self._handle_simple_query(user_query, query_type)
            state['current_step'] = 'completed'
            state['needs_more_research'] = False

        return state

    # 根据提问进行分类，决定下一步的动作，返回提问类型
    def _classify_query(self, user_query: str) -> str:
        prompt = self.prompt_loader.load(
            'coordinator_classify_query',
            user_query=user_query
        )

        query_type = self.llm.generate(prompt).strip().upper()
        if query_type not in ['GREETING', 'INAPPROPRIATE', 'RESEARCH']:
            query_type = 'RESEARCH'

        return query_type

    # 处理简单问题，返回回答
    def _handle_simple_query(self, user_query: str, query_type: str) -> str:
        prompt = self.prompt_loader.load(
            'coordinator_simple_response',
            user_query=user_query,
            query_type=query_type
        )
        response = self.llm.generate(prompt).strip()
        return response
