from alembic.testing.env import staging_env

from RAgents.agents.coordinator import Coordinator
from RAgents.agents.planner import Planner
from RAgents.agents.rapporteur import Rapporteur
from RAgents.agents.researcher import Researcher
from RAgents.workflow.state import ResearchState


class WorkflowNodes:
    def __init__(
            self,
            coordinator: Coordinator,
            planner: Planner,
            researcher: Researcher,
            rapporteur: Rapporteur
    ):
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur

    def coordinator_node(self, state: ResearchState) -> ResearchState:
        if state.get('query_type') in ['GREETING', 'INAPPROPRIATE']:
            state['current_step'] = 'completed'
            return state

        state['current_step'] = 'coordinating'
        state = self.coordinator.delegate_to_planner(state)
        return state

    def planner_node(self, state: ResearchState) -> ResearchState:
        state['current_step'] = 'planning'
        # 如果有用户反馈，则修改计划
        if state.get('user_feedback') and state.get('research_plan'):
            state = self.planner.modify_plan(state, state['user_feedback'])
        # 如果没有计划，则创建计划
        elif not state.get('research_plan'):
            state = self.planner.create_research_plan(state)
        return state

    # 人工审核节点，看当前计划是否存在问题
    def human_review_node(self, state: ResearchState) -> ResearchState:
        state['current_step'] = 'awaiting_approval'
        if state.get('auto_approve_plan', False):
            state['plan_approved'] = True
        return state

    def researcher_node(self, state: ResearchState) -> ResearchState:
        state['current_step'] = 'researching'
        next_task = self.planner.get_next_task(state)

        if next_task:
            state = self.researcher.execute_task(state, next_task)
            state['current_task'] = next_task
            state['iteration_count'] += 1
        else:
            state['needs_more_research'] = False
        return state

    def rapporteur_node(self, state: ResearchState) -> ResearchState:
        state['current_step'] = 'generating_report'
        state = self.rapporteur.generate_report(state)
        return state

    def should_continue_to_planner(self, state: ResearchState) -> str:
        if state.get('query_type') in ['GREETING', 'INAPPROPRIATE']:
            return "end"
        return "planner"

    def should_continue_research(self, state: ResearchState) -> str:
        if not state.get('plan_approved'):
            return "planner"
        return "researcher"

    def should_generate_report(self, state: ResearchState) -> str:
        if state['iteration_count'] >= state['max_iterations']:
            return "rapporteur" # 完成了所有迭代
        if self.planner.evaluate_context_sufficiency(state):
            return "rapporteur" # 上下文充分

        next_task = self.planner.get_next_task(state)
        if next_task:
            return "researcher"
        else:
            return "rapporteur"

    def should_start_conversation(self, state: ResearchState) -> str:
        if state.get('conversation_mode', False):
            return "conversation"
        return "end"

def create_node_functions(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur
) -> WorkflowNodes:
    return WorkflowNodes(coordinator, planner, researcher, rapporteur)