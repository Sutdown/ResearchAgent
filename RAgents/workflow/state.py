from typing import TypedDict, List, Annotated, Optional, Any, Literal
import operator

class ResearchState(TypedDict):
    # User query & meta
    query: str # 用户提问
    query_type: Literal["GREETING", "INAPPROPRIATE", "RESEARCH"]

    # Planning
    research_plan: Optional[dict] # 研究计划
    plan_approved: bool           # 计划是否已批准

    # Execution / research
    research_results: Annotated[list, operator.add] # 研究结果
    current_task: Optional[dict] # 当前任务
    iteration_count: int         # 迭代次数
    max_iterations: int          # 最大迭代次数
    needs_more_research: bool    # 是否需要更多研究

    # Reporting
    final_report: Optional[str] # 最终报告
    output_format: str          # 输出格式

    # UX / control flags
    current_step: str            # 当前步骤
    user_feedback: Optional[str] # 用户反馈
    auto_approve_plan: bool      # 自动批准计划
    simple_response: Optional[str] # 简单回答

# 研究计划结构
class PlanStructure(TypedDict):
    research_goal: str       # 研究目的
    sub_tasks: List[dict]    # 子任务
    completion_criteria: str # 完成标准
    estimated_iterations: int  # 预计迭代次数

# 子任务
class SubTask(TypedDict):
    task_id: int      # 任务ID
    description: str  # 任务描述
    search_queries: List[str]  # 搜索查询
    sources: List[str]         # 来源
    status: str                # 任务状态(pending, in_progress, completed)
    priority: Optional[int]    # 优先级

# 搜索结果
# 每个步骤的研究结构都是这个search research
# 最终得到的结果就是每个步骤中得到的结果的和
class SearchResult(TypedDict):
    task_id: int  # 任务ID
    query: str    # 搜索查询
    source: str   # 来源
    results: List[dict]  # 搜索结果
    timestamp: str       # 时间戳

# 单个结果
class IndividualResult(TypedDict):
    title: str         # 标题
    url: Optional[str] # 链接
    snippet: str       # 摘要
    relevance_score: Optional[float] # 相关性分数
    metadata: Optional[dict]         # 元信息