"""
简单的LangSmith观测功能实现
"""

import os
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()
try:
    from langsmith import Client, trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    trace = None


class SimpleLangSmithTracer:
    """简单的LangSmith追踪器"""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        self._initialize()
    
    def _initialize(self):
        """初始化LangSmith客户端"""
        if not LANGSMITH_AVAILABLE:
            print("LangSmith未安装，跳过观测功能")
            return
        
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            print("未设置LANGSMITH_API_KEY，跳过观测功能")
            return
        
        try:
            self.client = Client()
            self.enabled = True
            print("LangSmith追踪器已启用")
        except Exception as e:
            print(f"LangSmith初始化失败: {e}")
    
    def trace_agent(self, agent_name: str, operation: str):
        """代理操作追踪装饰器"""
        def decorator(func):
            if not self.enabled:
                return func
            
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    with trace(f"{agent_name}.{operation}"):
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        print(f"[{agent_name}] {operation} 完成，耗时: {duration:.2f}s")
                        return result
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"[{agent_name}] {operation} 失败，耗时: {duration:.2f}s，错误: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def trace_workflow(self, workflow_name: str):
        """工作流追踪装饰器"""
        def decorator(func):
            if not self.enabled:
                return func
            
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    with trace(f"workflow.{workflow_name}"):
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        print(f"[工作流] {workflow_name} 完成，耗时: {duration:.2f}s")
                        return result
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"[工作流] {workflow_name} 失败，耗时: {duration:.2f}s，错误: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def log_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """记录事件"""
        if not self.enabled:
            return
        
        try:
            with trace(f"event.{event_name}"):
                print(f"[事件] {event_name}")
                if data:
                    print(f"  数据: {data}")
        except Exception as e:
            print(f"事件记录失败: {e}")


# 全局追踪器实例
_tracer = None

def get_tracer() -> SimpleLangSmithTracer:
    """获取全局追踪器实例"""
    global _tracer
    if _tracer is None:
        _tracer = SimpleLangSmithTracer()
    return _tracer

def setup_langsmith_tracing():
    """设置LangSmith追踪"""
    tracer = get_tracer()
    print(f"LangSmith追踪状态: {'已启用' if tracer.enabled else '未启用'}")
    return tracer