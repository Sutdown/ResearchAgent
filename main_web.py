from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import gradio as gr
from dotenv import load_dotenv

# ===== 原有系统依赖 =====
from RAgents.agents.conversation import ConversationManager
from RAgents.agents.coordinator import Coordinator
from RAgents.agents.planner import Planner
from RAgents.agents.rapporteur import Rapporteur
from RAgents.agents.researcher import Researcher
from RAgents.llms.factory import LLMFactory
from RAgents.workflow.graph import ResearchWorkflow
from RAgents.utils.config import load_config_from_env
from RAgents.utils.logger import setup_logger
from RAgents.langsmith.langsmith import setup_langsmith_tracing

# ======================
# 全局状态（Web 专用）
# ======================
log_buffer: list[str] = []

approval_state = {
    "waiting": False,
    "approved": None,
    "feedback": None
}

final_report_holder = {
    "report": None
}

# ======================
# 工具函数
# ======================
def log(msg: str):
    log_buffer.append(msg)

def reset_state():
    log_buffer.clear()
    approval_state.update({
        "waiting": False,
        "approved": None,
        "feedback": None
    })
    final_report_holder["report"] = None

# ======================
# Web 版人工审批回调
# ======================
def human_approval_callback(state: Dict[str, Any]):
    log("\n🟡 等待人工审批...\n")
    approval_state["waiting"] = True

    while approval_state["approved"] is None:
        time.sleep(0.2)

    approval_state["waiting"] = False

    if approval_state["approved"]:
        log("✅ 研究计划已批准\n")
        return True, None
    else:
        feedback = approval_state["feedback"] or "请重新优化研究计划"
        log(f"❌ 计划被拒绝，反馈：{feedback}\n")
        return False, feedback

# ======================
# Web 研究执行函数（核心）
# ======================
def run_research_web(
    query: str,
    provider: str,
    model: str,
    max_iterations: int,
    auto_approve: bool,
    output_format: str
):
    reset_state()
    yield ""

    if not query.strip():
        yield "❌ 研究问题不能为空"
        return

    def task():
        try:
            setup_logger()
            setup_langsmith_tracing()
            load_dotenv()

            env_cfg = load_config_from_env()
            os.environ["LLM_PROVIDER"] = provider
            env_cfg = load_config_from_env()
            env_cfg.llm.model = model
            env_cfg.workflow.max_iterations = max_iterations
            env_cfg.workflow.auto_approve_plan = auto_approve

            log(f"🚀 使用模型：{provider.upper()} / {model}\n")

            llm = LLMFactory.create_llm(
                provider=env_cfg.llm.provider,
                api_key=env_cfg.llm.api_key,
                model=env_cfg.llm.model
            )

            coordinator = Coordinator(llm)
            planner = Planner(llm)

            researcher = Researcher(
                llm=llm,
                tavily_api_key=env_cfg.search.tavily_api_key,
                mcp_server_url=env_cfg.search.mcp_server_url,
                mcp_api_key=env_cfg.search.mcp_api_key,
                enable_vector_memory=False,
                vector_memory_path="./vector_memory"
            )

            def stream_callback(chunk: str):
                if chunk:
                    log(chunk)

            rapporteur = Rapporteur(llm, stream_callback=stream_callback)

            workflow = ResearchWorkflow(
                coordinator,
                planner,
                researcher,
                rapporteur,
                langsmith_config=env_cfg.langsmith
            )

            stream = workflow.stream_interactive(
                query=query,
                max_iterations=max_iterations,
                auto_approve=auto_approve,
                human_approval_callback=None if auto_approve else human_approval_callback,
                output_format=output_format
            )

            current_state = None

            for update in stream:
                for _, state in update.items():
                    if isinstance(state, dict):
                        current_state = state

            if current_state and current_state.get("final_report"):
                final_report_holder["report"] = current_state["final_report"]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("./outputs")
                out_dir.mkdir(exist_ok=True)

                suffix = "html" if output_format == "html" else "md"
                path = out_dir / f"research_{timestamp}.{suffix}"
                rapporteur.save_report(final_report_holder["report"], str(path))

                log(f"\n📄 报告已保存：{path}\n")

        except Exception as e:
            log(f"\n❌ 发生错误：{e}\n")

    threading.Thread(target=task, daemon=True).start()

    while True:
        time.sleep(0.3)
        yield "\n".join(log_buffer)

# ======================
# 审批按钮
# ======================
def approve_plan():
    approval_state["approved"] = True
    return "✅ 已批准"

def reject_plan(feedback):
    approval_state["approved"] = False
    approval_state["feedback"] = feedback
    return "❌ 已拒绝"

# ======================
# Gradio UI
# ======================
with gr.Blocks(title="Deep Research研究系统") as demo:
    gr.Markdown("# 🧠 Deep Research System")

    with gr.Row():
        query = gr.Textbox(label="研究问题", lines=3)
        provider = gr.Dropdown(
            ["deepseek", "openai", "claude", "gemini"],
            value="deepseek",
            label="LLM Provider"
        )

    model = gr.Textbox(label="模型名称", value="deepseek-chat")
    max_iter = gr.Slider(1, 10, value=5, step=1, label="最大迭代次数")
    auto = gr.Checkbox(label="自动批准研究计划", value=False)
    output_format = gr.Radio(["md", "html"], value="md", label="输出格式")

    start_btn = gr.Button("🚀 开始研究")

    log_box = gr.Textbox(
        label="运行日志（实时）",
        lines=20,
        interactive=False
    )

    gr.Markdown("## 👤 人工审批（仅在关闭自动批准时生效）")

    feedback_box = gr.Textbox(label="拒绝反馈（可选）")
    with gr.Row():
        approve_btn = gr.Button("✅ 批准")
        reject_btn = gr.Button("❌ 拒绝")

    approve_btn.click(approve_plan, outputs=log_box)
    reject_btn.click(reject_plan, inputs=feedback_box, outputs=log_box)

    start_btn.click(
        run_research_web,
        inputs=[query, provider, model, max_iter, auto, output_format],
        outputs=log_box
    )

demo.launch()
