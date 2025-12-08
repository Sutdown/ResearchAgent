import json
from typing import Dict, List, Optional
from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader
from RAgents.workflow.state import ResearchState, PlanStructure

class Planner:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_loader = PromptLoader()

    # 基于当前查询问题和用户反馈，创建研究计划
    def create_research_plan(self, state: ResearchState) -> ResearchState:
        query = state['query']
        user_feedback = state.get('user_feedback', '')
        prompt = self.prompt_loader.load(
            'planner_create_plan',
            query=query,
            user_feedback=user_feedback if user_feedback else None
        )
        response = self.llm.generate(prompt, temperature=0.7)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                plan = json.loads(json_str)
            else:
                plan = self._create_fallback_plan(query)

            allowed_sources = {"tavily"}
            for task in plan.get('sub_tasks', []):
                sources = task.get('sources') or ["tavily"]
                # filter out sources that are not allowed
                task['sources'] = [s for s in sources if s in allowed_sources] or ["tavily"]
                task['status'] = task.get('status', 'pending')

            state['research_plan'] = plan
            state['max_iterations'] = plan.get('estimated_iterations', 3)
        except json.JSONDecodeError:
            plan = self._create_fallback_plan(query)
            state['research_plan'] = plan

        return state

    def _create_fallback_plan(self, query: str) -> PlanStructure:
        return {
            'research_goal': query,
            'sub_tasks': [
                {
                    'task_id': 1,
                    'description': f'Research: {query}',
                    'search_queries': [query],
                    'sources': ['tavily'],
                    'status': 'pending',
                    'priority': 1
                }
            ],
            'completion_criteria': 'Gather sufficient information to answer the query',
            'estimated_iterations': 2
        }

    # 基于用户的修改意见修改计划
    def modify_plan(self, state: ResearchState, modifications: str) -> ResearchState:
        current_plan = state['research_plan']
        prompt = self.prompt_loader.load(
            'planner_modify_plan',
            current_plan=json.dumps(current_plan, indent=2),
            modifications=modifications
        )
        response = self.llm.generate(prompt, temperature=0.7)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                modified_plan = json.loads(json_str)
                state['research_plan'] = modified_plan
        except json.JSONDecodeError:
            pass

        return state