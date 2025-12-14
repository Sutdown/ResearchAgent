from __future__ import annotations

from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.panel import Panel

from RAgents.agents.conversation import ConversationManager
from RAgents.agents.coordinator import Coordinator
from RAgents.agents.planner import Planner
from RAgents.agents.rapporteur import Rapporteur
from RAgents.agents.researcher import Researcher
from RAgents.llms.factory import LLMFactory
from RAgents.utils.config import load_config_from_env
from RAgents.workflow.graph import ResearchWorkflow
from func import parse_args, print_welcome, print_menu, console, show_models, CLIConfig, \
    configure_settings, print_separator, get_api_key_for_provider
from RAgents.utils.logger import setup_logger
from RAgents.langsmith.langsmith import setup_langsmith_tracing


def human_approval_callback(state: Dict[str, Any]) -> tuple[bool, None] | tuple[bool, str]:
    console.print("\n")
    print_separator("=")
    console.print("[bold yellow]等待您的决策[/bold yellow]\n")

    console.print("[cyan]您可以选择：[/cyan]")
    console.print("  [green]1.[/green] 批准计划 - 开始执行研究")
    console.print("  [green]2.[/green] 拒绝计划 - 提供反馈重新制定")
    console.print("  [green]3.[/green] 取消任务 - 退出研究")
    console.print()

    choice = input("请选择操作 (1-3): ").strip()

    if choice == "1":
        # 批准计划
        console.print("[green]✓ 计划已批准，开始研究...[/green]\n")
        print_separator("=")
        return True, None

    elif choice == "2":
        # 拒绝并提供反馈
        console.print("\n[yellow]请提供修改意见（描述您希望如何调整研究计划）：[/yellow]")
        console.print("[dim]提示：您可以要求增加/删除某些研究方向，调整优先级等[/dim]\n")

        feedback = input("> ").strip()

        if not feedback:
            console.print("[yellow]未提供反馈，将重新生成计划...[/yellow]")
            feedback = "请重新优化研究计划"

        console.print(f"\n[cyan]已收到反馈，正在重新制定计划...[/cyan]\n")
        print_separator("=")
        return False, feedback

    elif choice == "3":
        # 取消任务
        console.print("\n[yellow]任务已取消[/yellow]")
        raise KeyboardInterrupt("用户取消任务")

    else:
        # 无效选择，默认拒绝
        console.print("[red]无效选择，请重新决策[/red]")
        return human_approval_callback(state)

def execute_conversation(config: CLIConfig) -> None:
    # 执行多轮对话
    global logger
    print_separator("-")
    console.print("[bold cyan]多轮对话模式[/bold cyan]\n")

    try:
        logger = setup_logger()
        # 加载配置
        console.print("\n[dim]正在加载配置...[/dim]")
        env_cfg = load_config_from_env()
        # Override with CLI config
        os.environ['LLM_PROVIDER'] = config.provider
        env_cfg = load_config_from_env()  # Reload
        env_cfg.llm.model = config.model

        console.print(f"[dim]正在初始化 {config.provider.upper()} LLM...[/dim]")
        llm = LLMFactory.create_llm(
            provider=env_cfg.llm.provider,
            api_key=env_cfg.llm.api_key,
            model=env_cfg.llm.model
        )

        console.print("[dim]正在初始化多轮对话管理器...[/dim]")

        conversation_config = {
            'llm_provider': env_cfg.llm.provider,
            'llm_api_key': env_cfg.llm.api_key,
            'llm_model': env_cfg.llm.model,
            'tavily_api_key': env_cfg.search.tavily_api_key,
            'mcp_server_url': env_cfg.search.mcp_server_url,
            'mcp_api_key': env_cfg.search.mcp_api_key,
            'vector_memory_path': './vector_memory'  # Use default path
        }
        conversation_manager = ConversationManager(conversation_config)

        success = conversation_manager.start_conversation()
        if success:
            console.print("[green]✓ 多轮对话已正常结束[/green]")
        else:
            console.print("[yellow]多轮对话被中断[/yellow]")
    except KeyboardInterrupt:
        console.print("\n\n[yellow]多轮对话已被用户中断[/yellow]")
        print_separator("-")
    except Exception as e:
        console.print(f"\n[red]✗ 发生错误：{e}[/red]")
        logger.exception("Conversation error")
        print_separator("-")

def execute_research(config: CLIConfig, query: str = None) -> None:
    global logger
    print_separator("-")
    console.print("[bold cyan]执行研究任务[/bold cyan]\n")

    if not query:
        query = input("请输入研究问题：\n> ").strip()
    if not query:
        console.print("[red]✗ 研究问题不能为空[/red]")
        return

    try:
        logger = setup_logger()
        # 加载配置
        console.print("\n[dim]正在加载配置...[/dim]")
        env_cfg = load_config_from_env()

        os.environ['LLM_PROVIDER'] = config.provider
        env_cfg = load_config_from_env()  # Reload
        env_cfg.llm.model = config.model
        # 根据运行模式调整工作流参数
        if config.mode == "fast":
            # 快速模式：限制最大迭代次数
            env_cfg.workflow.max_iterations = min(config.max_iterations, 3)
        else:
            env_cfg.workflow.max_iterations = config.max_iterations
        env_cfg.workflow.auto_approve_plan = config.auto_approve

        # 创建工作流, 执行研究
        console.print(f"[dim]正在初始化 {config.provider.upper()} LLM...[/dim]")
        llm = LLMFactory.create_llm(
            provider=env_cfg.llm.provider,
            api_key=env_cfg.llm.api_key,
            model=env_cfg.llm.model
        )

        console.print("[dim]正在初始化智能体...[/dim]")
        coordinator = Coordinator(llm)
        planner = Planner(llm)

        enable_vector_memory = env_cfg.langsmith.enabled if env_cfg.langsmith else False
        researcher = Researcher(
            llm=llm,
            tavily_api_key=env_cfg.search.tavily_api_key,
            mcp_server_url=env_cfg.search.mcp_server_url,
            mcp_api_key=env_cfg.search.mcp_api_key,
            enable_vector_memory=enable_vector_memory,
            vector_memory_path="./vector_memory"
        )
        if config.mode == "fast":
            researcher.max_requests_per_task = 2

        # 定义一个用于报告阶段的流式输出回调
        def stream_printer(chunk: str) -> None:
            if not chunk:
                return
            # 使用 rich 控制台直接追加文本，不自动换行
            console.print(chunk, end="")
        rapporteur = Rapporteur(llm, stream_callback=stream_printer)

        console.print("[dim]正在设置研究工作流...[/dim]\n")
        workflow = ResearchWorkflow(
            coordinator,
            planner,
            researcher,
            rapporteur,
            langsmith_config=env_cfg.langsmith
        )

        # 运行workflow
        print_separator("-")
        console.print(f"[bold green]开始研究：[/bold green]{query}\n")
        current_state = None

        stream_iter = workflow.stream_interactive(
            query,
            config.max_iterations,
            auto_approve=config.auto_approve,
            human_approval_callback=human_approval_callback if not config.auto_approve else None,
            output_format=config.output_format
        )

        for state_update in stream_iter:
            if config.show_steps:
                console.print(f"[dim]state_update type: {type(state_update)}[/dim]")

            for node_name, state in state_update.items():
                if config.show_steps:
                    console.print(f"[dim]node: {node_name}, state type: {type(state)}[/dim]")

                # 检查当前状态是否为元组或字典
                if isinstance(state, tuple):
                    if len(state) >= 1:
                        current_state = state[0] if isinstance(state[0], dict) else state
                    else:
                        continue
                else:
                    current_state = state

                # 检查当前状态是否为字典
                if not isinstance(current_state, dict):
                    if config.show_steps:
                        console.print(f"[yellow]Warning: state is not dict: {type(current_state)}[/yellow]")
                    continue

                # 查看当前状态
                step = current_state.get('current_step', 'unknown')

                if config.show_steps:
                    console.print(f"[magenta]步骤：{step}[/magenta]")

                if current_state.get('simple_response'):
                    console.print(f"\n{current_state['simple_response']}\n")
                    current_state = current_state  # Store for later
                    continue

                if step == 'planning':
                    console.print("[cyan]正在创建研究计划...[/cyan]")
                    if current_state.get('research_plan'):
                        plan_display = planner.format_plan_for_display(current_state['research_plan'])
                        console.print(Panel(plan_display, title="研究计划", border_style="blue"))
                elif step == 'awaiting_approval':
                    if config.auto_approve:
                        console.print("[green]✓ 计划已自动批准[/green]")
                elif step == 'researching':
                    task = current_state.get('current_task', {})
                    iteration = current_state.get('iteration_count', 0)
                    console.print(f"[cyan]正在研究：{task.get('description', '未知任务')}[/cyan]")
                    console.print(f"[dim]迭代 {iteration}/{config.max_iterations}[/dim]")
                elif step == 'generating_report':
                    console.print("[cyan]正在生成最终报告...[/cyan]")

        if current_state and current_state.get('final_report'):
            report = current_state['final_report']

            console.print("\n")
            console.print(Panel(
                Markdown(report),
                title="研究报告",
                border_style="green"
            ))

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            file_extension = 'html' if current_state.get('output_format') == 'html' else 'md'
            output_path = output_dir / f"research_report_{timestamp}.{file_extension}"
            rapporteur.save_report(report, str(output_path))
            console.print(f"\n[green]✓ 报告已保存至：{output_path}[/green]")

        elif current_state and current_state.get('simple_response'):
            pass
        else:
            console.print("[red]✗ 研究未成功完成[/red]")

        print_separator("-")
    except KeyboardInterrupt:
        console.print("\n\n[yellow]任务已被用户中断[/yellow]")
        print_separator("-")
    except Exception as e:
        console.print(f"\n[red]✗ 发生错误：{e}[/red]")
        logger.exception("Research error")
        print_separator("-")

def interactive_mode(config: CLIConfig) -> int:
    print_welcome()
    try:
        while True:
            try:
                print_menu()
                choice = input("请选择操作 (1-6): ").strip()

                if choice == "1":
                    # 执行研究任务
                    execute_research(config)

                elif choice == "2":
                    # 多轮对话
                    execute_conversation(config)

                elif choice == "3":
                    # 查看可用模型
                    console.print("\n[bold]选择 LLM 提供商：[/bold]\n")
                    console.print("  [cyan]1[/cyan] - DeepSeek")
                    console.print("  [cyan]2[/cyan] - OpenAI")
                    console.print("  [cyan]3[/cyan] - Claude")
                    console.print("  [cyan]4[/cyan] - Gemini")

                    provider_choice = input("\n选择提供商 (1-4): ").strip()
                    provider_map = {'1': 'deepseek', '2': 'openai', '3': 'claude', '4': 'gemini'}
                    provider = provider_map.get(provider_choice)

                    if provider:
                        show_models(provider)
                    else:
                        console.print("[red]✗ 无效的选择[/red]")

                elif choice == "4":
                    # 配置设置
                    configure_settings(config)

                elif choice == "5":
                    # 查看当前配置
                    print_separator("-")
                    console.print("[bold cyan]当前配置：[/bold cyan]\n")
                    console.print(f"  提供商：[yellow]{config.provider}[/yellow]")
                    console.print(f"  模型：[yellow]{config.model}[/yellow]")
                    console.print(f"  最大迭代次数：[yellow]{config.max_iterations}[/yellow]")
                    console.print(f"  自动批准：[yellow]{'是' if config.auto_approve else '否'}[/yellow]")
                    console.print(f"  输出目录：[yellow]{config.output_dir}[/yellow]")
                    console.print(f"  输出格式：[yellow]{config.output_format.upper()}[/yellow]")
                    console.print(f"  显示步骤：[yellow]{'是' if config.show_steps else '否'}[/yellow]")
                    console.print()
                    print_separator("-")

                elif choice == "6":
                    # 退出程序
                    console.print("\n[yellow]感谢使用 SDYJ 深度研究系统！再见！[/yellow]\n")
                    return 0

                else:
                    console.print("[red]✗ 无效的选择，请输入 1-6[/red]")

            except KeyboardInterrupt:
                console.print("\n\n[yellow]感谢使用！再见！[/yellow]\n")
                return 0
            except EOFError:
                console.print("\n\n[yellow]感谢使用！再见！[/yellow]\n")
                return 0
            except Exception as e:
                console.print(f"\n[red]✗ 发生错误：{e}[/red]\n")

    except Exception as e:
        console.print(f"\n[red]✗ 系统错误：{e}[/red]\n")
        return 1

def main(argv: Any = None) -> int:
    """主入口函数"""
    load_dotenv()
    
    # 初始化LangSmith追踪
    setup_langsmith_tracing()
    
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # 检查 API 密钥
    api_key = get_api_key_for_provider(args.provider)
    if not api_key:
        print(f"缺少 API 密钥")
        print(f"请在 .env 文件中设置API_KEY")
        return 2

    # 如果没有指定模型，使用默认模型
    if not args.model:
        model_defaults = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4',
            'claude': 'claude-3-5-sonnet-20241022',
            'gemini': 'gemini-pro'
        }
        args.model = model_defaults.get(args.provider, 'deepseek-chat')

    # 创建配置
    config = CLIConfig(
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        auto_approve=args.auto_approve,
        output_dir=args.output_dir,
        show_steps=args.show_steps,
        output_format=args.output_format,
        mode=args.mode,
    )

    # 如果指定了交互模式或没有提供任务，进入交互式菜单
    if args.interactive or not args.query:
        return interactive_mode(config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())