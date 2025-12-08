import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# 初始化日志系统，程序启动时调用
def setup_logger(
    name: str = "Research_Agents",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name) # 同名logger是全局单例，共享相同配置
    logger.setLevel(level)
    logger.handlers = []

    # console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # 没有使用rich时，使用默认的格式化器
    if not use_rich:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file) # 创建日志文件所在的目录
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 设置file handler的格式
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str = "Research_Agents") -> logging.Logger:
    return logging.getLogger(name)

# 日志能力模块，允许多继承
class LoggerMixin:
    @property # 把函数变成只读属性
    def logger(self) -> logging.Logger:
        name = f"Research_Agents.{self.__class__.__name__}"
        return logging.getLogger(name)

# 用户提示
console = Console()
def print_success(message: str):
    console.print(f"[green]✓[/green] {message}")

def print_error(message: str):
    console.print(f"[red]✗[/red] {message}")

def print_warning(message: str):
    console.print(f"[yellow]⚠[/yellow] {message}")

def print_info(message: str):
    console.print(f"[blue]ℹ[/blue] {message}")

def print_step(message: str):
    console.print(f"[cyan]▶[/cyan] {message}")