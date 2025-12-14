from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from RAgents.agents.coordinator import Coordinator
from RAgents.agents.planner import Planner
from RAgents.agents.rapporteur import Rapporteur
from RAgents.agents.researcher import Researcher
from RAgents.workflow.nodes import WorkflowNodes
from RAgents.workflow.state import ResearchState
from RAgents.langsmith.langsmith import get_tracer


def create_research_graph(
    coordinator: Coordinator,
    planner: Planner,
    researcher: Researcher,
    rapporteur: Rapporteur,
    langsmith_config=None
):
    nodes = WorkflowNodes(coordinator, planner, researcher, rapporteur)
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("coordinator", nodes.coordinator_node)
    workflow.add_node("planner", nodes.planner_node)
    workflow.add_node("human_review", nodes.human_review_node)
    workflow.add_node("researcher", nodes.researcher_node)
    workflow.add_node("rapporteur", nodes.rapporteur_node)

    workflow.add_edge(START, "coordinator")

    # 是否为需要计划的问题，如果不是，则直接结束
    workflow.add_conditional_edges(
        "coordinator",
        nodes.should_continue_to_planner,
        {
            "planner": "planner",  # Research query
            "end": END  # Simple query (greeting/inappropriate)
        }
    )
    workflow.add_edge("planner", "human_review")
    # 对当前计划是否满意，不满意重新规划，满意则开始研究
    workflow.add_conditional_edges(
        "human_review",
        nodes.should_continue_research,
        {
            "planner": "planner",  # User wants modifications
            "researcher": "researcher"  # User approved, start research
        }
    )
    # 看当前研究是否完成
    workflow.add_conditional_edges(
        "researcher",
        nodes.should_generate_report,
        {
            "researcher": "researcher",  # Continue research
            "rapporteur": "rapporteur"  # Generate report
        }
    )
    workflow.add_edge("rapporteur", END)

    checkpointer = MemorySaver() # 添加检查点支持，保持工作流状态
    compile_kwargs = {
        "checkpointer": checkpointer,
        "interrupt_before": ["human_review"]
    }

    return workflow.compile(**compile_kwargs)


class ResearchWorkflow:
    def __init__(
            self,
            coordinator: Coordinator, planner: Planner,
            researcher: Researcher, rapporteur: Rapporteur,
            langsmith_config=None
    ):
        self.coordinator = coordinator
        self.planner = planner
        self.researcher = researcher
        self.rapporteur = rapporteur
        self.tracer = get_tracer()
        self.graph = create_research_graph(
            coordinator, planner, researcher, rapporteur, langsmith_config
        )

    @get_tracer().trace_workflow("research_workflow")
    def stream_interactive(
            self,
            query: str,
            max_iterations: Optional[int] = None,
            auto_approve: bool = False,
            human_approval_callback=None,
            output_format: str = "markdown"
    ):
        initial_state = self.coordinator.initialize_research(query, auto_approve=auto_approve,
                                                             output_format=output_format) # 初始化状态
        if max_iterations:
            initial_state['max_iterations'] = max_iterations # 设置最大迭代次数
        config = {"configurable": {"thread_id": "1"}} # langgraph需要线程配置支持检查点功能
        approval_handled = False # 是否已经处理了审批

        for output in self.graph.stream(initial_state, config=config):
            yield output # 暂停函数执行，返回当前节点结果给调用者
            if "__interrupt__" in output and not approval_handled:
                # 提取工作流快照和当前状态
                current_snapshot = self.graph.get_state(config)
                current_state = current_snapshot.values

                if isinstance(current_state, dict) and current_state.get('research_plan'):
                    if auto_approve: # 自动批准计划
                        current_state['plan_approved'] = True
                        current_state['user_feedback'] = None
                        self.graph.update_state(config, current_state)
                        # 人工处理
                    elif human_approval_callback and not current_state.get('plan_approved', False):
                        current_state['current_step'] = 'awaiting_approval'
                        approved, feedback = human_approval_callback(current_state) # 执行函数等待人工反馈
                        if approved:
                            current_state['plan_approved'] = True
                            current_state['user_feedback'] = None
                        else:
                            current_state['plan_approved'] = False
                            current_state['user_feedback'] = feedback
                        self.graph.update_state(config, current_state)
                    approval_handled = True

                for continue_output in self.graph.stream(None, config=config):
                    yield continue_output # 继续执行剩余的工作流
                return

    def visualize(self, output_path: Optional[str] = None) -> str:
        try:
            from langgraph.graph import Graph
            mermaid = self.graph.get_graph().draw_mermaid()
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(mermaid)
                return output_path
            else:
                return mermaid
        except Exception as e:
            return f"Visualization not available: {str(e)}"