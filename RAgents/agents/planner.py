import json
from typing import Dict, List, Optional
from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader
from RAgents.workflow.state import ResearchState, PlanStructure, SubTask


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

    # 评估上下文的充分性
    def evaluate_context_sufficiency(self, state: ResearchState) -> bool:
        query = state['query']
        plan = state['research_plan']
        results = state['research_results']
        iteration = state['iteration_count']
        max_iterations = state['max_iterations']

        # 当迭代次数满足要求时，已经有了足够的信息
        if iteration >= max_iterations:
            return True
        # 当迭代次数不满足要求时，没有足够的信息
        if not results:
            return False
        # 考虑到迭代次数和结果数量基本满足要求
        if iteration >= 2 and len(results) >= 10:
            return True

        prompt = self.prompt_loader.load(
            'planner_evaluate_context',
            query=query,
            research_goal=plan.get('research_goal', query),
            completion_criteria=plan.get('completion_criteria', 'N/A'),
            results_count=len(results),
            current_iteration=iteration + 1,
            max_iterations=max_iterations
        )

        response = self.llm.generate(prompt, temperature=0.3).strip().upper()
        return response == "YES"

    # 得到下一步任务
    def get_next_task(self, state: ResearchState) -> Optional[SubTask]:
        plan = state.get('research_plan')
        if not plan:
            return None

        tasks = sorted(
            plan.get('sub_tasks', []),
            key=lambda t: (t.get('priority', 99), t.get('task_id', 0))
        )
        for task in tasks:
            if task.get('status') == 'pending':
                return task

        return None

    # 格式化当前计划进行展示
    def format_plan_for_display(self, plan: PlanStructure) -> str:
        output = []
        output.append(f"研究目标: {plan.get('research_goal', 'N/A')}")
        output.append(f"\n预计迭代次数: {plan.get('estimated_iterations', 'N/A')}")
        output.append(f"\n完成标准: {plan.get('completion_criteria', 'N/A')}")
        output.append("\n\n子任务列表:")
        for task in plan.get('sub_tasks', []):
            output.append(f"\n  {task['task_id']}. {task['description']}")
            output.append(f"\n     Queries: {', '.join(task.get('search_queries', []))}")
            output.append(f"\n     Sources: {', '.join(task.get('sources', []))}")
            output.append(f"\n     Priority: {task.get('priority', 'N/A')}")
            output.append(f"\n     Status: {task.get('status', 'pending')}")
        return ''.join(output)