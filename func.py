import argparse
import os
from dataclasses import dataclass
from typing import Any
from rich.console import Console
from rich.panel import Panel

@dataclass
class CLIConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    max_iterations: int = 5
    auto_approve: bool = False
    output_dir: str = "./outputs"
    show_steps: bool = False
    output_format: str = "markdown"  # "markdown" or "html"
    mode: str = "fast"  # "fast" or "full"

console = Console()

def configure_settings(config: CLIConfig) -> None:
    """配置设置"""
    print_separator("-")
    console.print("[bold cyan]当前配置：[/bold cyan]\n")
    console.print(f"  提供商：[yellow]{config.provider}[/yellow]")
    console.print(f"  模型：[yellow]{config.model}[/yellow]")
    console.print(f"  最大迭代次数：[yellow]{config.max_iterations}[/yellow]")
    console.print(f"  自动批准计划：[yellow]{'是' if config.auto_approve else '否'}[/yellow]")
    console.print(f"  输出目录：[yellow]{config.output_dir}[/yellow]")
    console.print(f"  输出格式：[yellow]{config.output_format.upper()}[/yellow]")
    console.print(f"  显示步骤：[yellow]{'是' if config.show_steps else '否'}[/yellow]")
    console.print(f"  运行模式：[yellow]{config.mode}[/yellow] (fast/full)")
    console.print()

    console.print("[cyan]选择要修改的设置（直接回车跳过）：[/cyan]\n")

    config_changed = False

    # 修改提供商
    provider_input = input(f"LLM 提供商 (deepseek/openai/claude/gemini) [{config.provider}]: ").strip().lower()
    if provider_input and provider_input in ["deepseek", "openai", "claude", "gemini"]:
        if provider_input != config.provider:
            # 检查 API 密钥
            new_api_key = get_api_key_for_provider(provider_input)
            if not new_api_key:
                console.print(f"[red]✗ 未找到 {provider_input.upper()}_API_KEY 环境变量[/red]")
                console.print(f"[yellow]请在 .env 文件中配置 {provider_input.upper()}_API_KEY[/yellow]")
            else:
                config.provider = provider_input
                # 自动更新默认模型
                model_defaults = {
                    'deepseek': 'deepseek-chat',
                    'openai': 'gpt-4',
                    'claude': 'claude-3-5-sonnet-20241022',
                    'gemini': 'gemini-pro'
                }
                config.model = model_defaults.get(provider_input, config.model)
                config_changed = True
                console.print(f"[green]✓ 已更新提供商为 {provider_input}，模型自动调整为 {config.model}[/green]")
    elif provider_input and provider_input not in ["deepseek", "openai", "claude", "gemini"]:
        console.print("[red]✗ 无效的提供商[/red]")

    # 修改模型
    model_input = input(f"模型名称 [{config.model}]: ").strip()
    if model_input:
        config.model = model_input
        config_changed = True
        console.print(f"[green]✓ 已更新模型为 {model_input}[/green]")

    # 修改最大迭代次数
    try:
        max_iter_input = input(f"最大迭代次数 [{config.max_iterations}]: ").strip()
        if max_iter_input:
            new_max_iter = int(max_iter_input)
            if new_max_iter > 0:
                config.max_iterations = new_max_iter
                config_changed = True
                console.print(f"[green]✓ 已更新最大迭代次数为 {new_max_iter}[/green]")
            else:
                console.print("[red]✗ 最大迭代次数必须大于 0[/red]")
    except ValueError:
        console.print("[red]✗ 无效的数字[/red]")

    # 修改自动批准
    auto_approve_input = input(f"自动批准计划 (y/n) [{'y' if config.auto_approve else 'n'}]: ").strip().lower()
    if auto_approve_input in ['y', 'yes', '是']:
        if not config.auto_approve:
            config.auto_approve = True
            config_changed = True
        console.print("[green]✓ 已启用自动批准[/green]")
    elif auto_approve_input in ['n', 'no', '否']:
        if config.auto_approve:
            config.auto_approve = False
            config_changed = True
        console.print("[green]✓ 已禁用自动批准[/green]")

    # 修改输出目录
    output_dir_input = input(f"输出目录 [{config.output_dir}]: ").strip()
    if output_dir_input:
        config.output_dir = output_dir_input
        config_changed = True
        console.print(f"[green]✓ 已更新输出目录为 {output_dir_input}[/green]")

    # 修改输出格式
    output_format_input = input(f"输出格式 (markdown/html) [{config.output_format}]: ").strip().lower()
    if output_format_input in ['markdown', 'md', 'html']:
        # 规范化格式名称
        normalized_format = 'markdown' if output_format_input in ['markdown', 'md'] else 'html'
        if normalized_format != config.output_format:
            config.output_format = normalized_format
            config_changed = True
            console.print(f"[green]✓ 已更新输出格式为 {normalized_format.upper()}[/green]")
    elif output_format_input:
        console.print("[red]✗ 无效的输出格式，请选择 markdown 或 html[/red]")

    # 修改显示步骤
    show_steps_input = input(f"显示步骤 (y/n) [{'y' if config.show_steps else 'n'}]: ").strip().lower()
    if show_steps_input in ['y', 'yes', '是']:
        if not config.show_steps:
            config.show_steps = True
            config_changed = True
        console.print("[green]✓ 已启用显示步骤[/green]")
    elif show_steps_input in ['n', 'no', '否']:
        if config.show_steps:
            config.show_steps = False
            config_changed = True
        console.print("[green]✓ 已禁用显示步骤[/green]")

    # 修改运行模式
    mode_input = input(f"运行模式 (fast/full) [{config.mode}]: ").strip().lower()
    if mode_input in ["fast", "full"]:
        if mode_input != config.mode:
            config.mode = mode_input
            config_changed = True
            console.print(f"[green]✓ 已更新运行模式为 {mode_input}[/green]")
    elif mode_input:
        console.print("[red]✗ 无效的运行模式，请选择 fast 或 full[/red]")

    print_separator("-")

def get_api_key_for_provider(provider: str) -> str | None:
    """根据提供商获取对应的 API 密钥"""
    provider_env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = provider_env_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None

def show_models(provider: str) -> None:
    """显示可用模型列表"""
    models = {
        'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'claude': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
        'gemini': ['gemini-pro', 'gemini-1.5-pro'],
        'deepseek': ['deepseek-chat', 'deepseek-coder']
    }

    print_separator("-")
    console.print(f"\n[bold cyan]{provider.upper()} 的可用模型：[/bold cyan]\n")

    for model in models.get(provider, []):
        console.print(f"  • {model}")
    console.print()
    print_separator("-")

def print_separator(char: str = "─", length: int = 70) -> None:
    """打印分隔线"""
    console.print(f"[cyan]{char * length}[/cyan]")

def parse_args(argv: Any) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepResearch系统 - 基于 LangGraph 的多智能体研究系统"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="研究问题或主题（可选，不提供则进入交互模式）"
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "openai", "claude", "gemini"],
        help="LLM 提供商（默认：deepseek）"
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="模型名称（默认根据提供商选择）"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="最大研究迭代次数（默认：5）"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=False,
        help="自动批准研究计划"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="报告输出目录（默认：./outputs）"
    )
    parser.add_argument(
        "--output-format",
        default="markdown",
        choices=["markdown", "html"],
        help="报告输出格式（默认：markdown）"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        default=False,
        help="显示详细执行步骤"
    )
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "full"],
        help="运行模式：fast 为快速模式，full 为完整模式（默认：fast）"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式菜单模式"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Deep Research System 0.1.0"
    )
    return parser.parse_args(argv)

def print_welcome() -> None:
    """打印欢迎界面"""
    console.print("\n")
    print_header("SDYJ 深度研究系统")
    console.print("[yellow]欢迎使用基于 LangGraph 的多智能体研究系统！[/yellow]")

def print_header(text: str) -> None:
    """打印标题"""
    console.print(Panel.fit(
        f"[bold cyan]{text}[/bold cyan]",
        border_style="cyan"
    ))

def print_menu() -> None:
    """打印主菜单"""
    console.print("\n[bold cyan]主菜单：[/bold cyan]\n")
    console.print("  [green]1.[/green] 执行研究任务")
    console.print("  [green]2.[/green] 多轮对话")
    console.print("  [green]3.[/green] 查看可用模型")
    console.print("  [green]4.[/green] 配置设置")
    console.print("  [green]5.[/green] 查看当前配置")
    console.print("  [green]6.[/green] 退出程序")
    console.print()