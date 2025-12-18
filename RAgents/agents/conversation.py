from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
import re

from RAgents.llms.factory import LLMFactory
from RAgents.prompts.loader import PromptLoader
from RAgents.tools.arxiv_search import ArxivSearch
from RAgents.tools.mcp_client import MCPClient
from RAgents.tools.tavily_search import TavilySearch
from RAgents.utils.vector import VectorMemory

console = Console()

class ConversationManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_history = []
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 创建LLM实例
        self.llm = LLMFactory.create_llm(
            provider=config['llm_provider'],
            api_key=config['llm_api_key'],
            model=config['llm_model']
        )
        self.prompt_loader = PromptLoader()
        # 创建工具实例
        self.tavily = TavilySearch(config.get('tavily_api_key')) if config.get('tavily_api_key') else None
        self.arxiv = ArxivSearch()
        self.mcp = MCPClient(config.get('mcp_server_url'), config.get('mcp_api_key')) if config.get('mcp_server_url') else None
        # 创建向量库实例
        self.vector_memory = VectorMemory(persist_directory=config.get('vector_memory_path', './vector_memory'))
        # 对话长度管理
        self.context_window = 5 # 上下文窗口大小
        self.relevance_threshold = 0.8 # 相似度阈值

    def start_conversation(self) -> bool | None:
        console.print("\n[bold cyan]🤖 Deep-Research多轮对话系统[/bold cyan]")
        console.print("[yellow]我可以基于历史研究报告和可用工具与您对话[/yellow]")
        console.print("[dim]输入 'exit' 或 'quit' 或 '退出' 结束对话[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold blue]用户:[/bold blue] ").strip()
                if self._is_exit_command(user_input):
                    console.print("\n[yellow]感谢使用！[/yellow]\n")
                    return True
                if not user_input:
                    continue
                self._process_user_input(user_input) # 处理用户输入
            except KeyboardInterrupt:
                console.print("\n\n[yellow]对话被中断[/yellow]\n")
                return False
            except Exception as e:
                console.print(f"[red]对话发生错误：{e}[/red]")

    def _is_exit_command(self, user_input: str) -> bool:
        exit_commands = ['exit', 'quit', '退出', '结束', 'bye', 'goodbye']
        return user_input.lower().strip() in exit_commands

    def _process_user_input(self, user_input: str) -> None:
        # 将用户输入添加到对话历史中
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # 处理用户输入
        try:
            intent = self._analyze_intent(user_input)

            if intent == 'simple_search':
                # 直接搜索
                response = self._handle_direct_search(user_input)
            elif intent == 'complex_research':
                # 复杂研究
                response = self._handle_complex_research(user_input)
            else:
                # 默认对话模式
                response = self._handle_conversation_query(user_input)

            console.print(f"[bold green]系统:[/bold green] {response}")
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            self.conversation_history.append({
                'role': 'assistant',
                'content': error_msg,
                'timestamp': datetime.now().isoformat()
            })

    # 分析用户意图
    def _analyze_intent(self, user_input: str) -> str:
        """分析用户意图"""
        # Simple keywords for search
        search_keywords = [
            '搜索', 'search', '查找', 'find', '最新', 'latest',
            '新闻', 'news', '论文', 'paper', '研究', 'research'
        ]
        # Complex research indicators
        research_indicators = [
            '分析', 'analyze', '详细研究', 'detailed research',
            '全面', 'comprehensive', '深入', 'in-depth'
        ]
        input_lower = user_input.lower()

        if any(keyword in input_lower for keyword in search_keywords):
            if any(indicator in input_lower for indicator in research_indicators):
                return 'complex_research'
            return 'simple_search'

        if any(indicator in input_lower for indicator in research_indicators):
            return 'complex_research'

        return 'conversation'

    # 处理直接搜索，有明确的搜索问题的情况，采用工具搜索
    def _handle_direct_search(self, user_input: str) -> str:
        search_query = self._extract_search_query(user_input)

        if not search_query:
            return "请提供更明确的搜索内容。"

        if self.tavily:
            try:
                results = self.tavily.search(search_query, max_results=3)
                if results and results.get('results'):
                    # Format results for display
                    formatted_results = []
                    for i, result in enumerate(results['results'], 1):
                        title = result.get('title', '无标题')
                        snippet = result.get('snippet', '无摘要')
                        url = result.get('url', '')

                        formatted_results.append(f"{i}. {title}")
                        formatted_results.append(f"   {snippet[:150]}...")
                        if url:
                            formatted_results.append(f"   链接: {url}")
                        formatted_results.append("")

                    return "\n".join(formatted_results)
                else:
                    return f"搜索 '{search_query}' 未找到相关结果。"

            except Exception as e:
                return f"搜索时出错: {str(e)}"

        # Fallback to arxiv if tavily is not available
        elif self.arxiv:
            try:
                results = self.arxiv.search(search_query, max_results=2)

                if results and results.get('results'):
                    # Format results for display
                    formatted_results = []
                    for i, result in enumerate(results['results'], 1):
                        title = result.get('title', '无标题')
                        snippet = result.get('snippet', '无摘要')
                        url = result.get('url', '')

                        formatted_results.append(f"{i}. {title}")
                        formatted_results.append(f"   {snippet[:150]}...")
                        if url:
                            formatted_results.append(f"   链接: {url}")
                        formatted_results.append("")

                    return f"从 arXiv 找到以下论文:\n\n" + "\n".join(formatted_results)
                else:
                    return f"搜索 '{search_query}' 未找到相关论文。"

            except Exception as e:
                return f"搜索时出错: {str(e)}"

        else:
            return "抱歉，当前没有可用的搜索工具。"

    def _extract_search_query(self, user_input: str) -> str:
        cleaned = re.sub(r'(搜索|search|查找|find|关于|about)[：:\s]*', '', user_input)
        cleaned = re.sub(r'[？?！!。.]$', '', cleaned)
        return cleaned.strip()

    # 处理复杂研究，会走主流程
    def _handle_complex_research(self, user_input: str) -> str:
        console.print("[yellow]正在进行深度研究，这可能需要一些时间...[/yellow]")
        try:
            from RAgents.agents.coordinator import Coordinator
            from RAgents.agents.planner import Planner
            from RAgents.agents.researcher import Researcher

            coordinator = Coordinator(self.llm)
            planner = Planner(self.llm)
            researcher = Researcher(
                llm=self.llm,
                tavily_api_key=self.config.get('tavily_api_key'),
                mcp_server_url=self.config.get('mcp_server_url'),
                mcp_api_key=self.config.get('mcp_api_key'),
                enable_vector_memory=True,
                vector_memory_path=self.config.get('vector_memory_path', './vector_memory')
            )

            state = coordinator.initialize_research(
                user_input,
                auto_approve=True,  # Auto-approve for conversation mode
                output_format="markdown"
            )
            if state.get('simple_response'):
                return state['simple_response']
            state = planner.create_research_plan(state)
            next_task = planner.get_next_task(state)
            if next_task:
                state = researcher.execute_task(state, next_task)
                relevant_info = researcher.extract_relevant_info(state)

                # Store results in vector memory
                if state.get('research_results'):
                    self.vector_memory.store_research_result(
                        query=user_input,
                        results={'search_results': state['research_results']},
                        quality_score=0.0,  # Will be updated based on feedback
                        metadata={
                            'session_id': self.session_id,
                            'conversation_mode': True
                        }
                    )
                response = (
                    f"我已开始研究 '{user_input}'，以下是初步发现:\n\n"
                    f"{relevant_info}\n\n"
                    "如果您需要更详细的研究，请使用完整的研究模式。"
                )
                return response
            else:
                return "无法为您的查询制定研究计划，请尝试重新表述或使用更具体的描述。"

        except Exception as e:
            return f"执行研究时出错: {str(e)}"

    # 处理默认多轮对话，采用短期记忆和向量库
    def _handle_conversation_query(self, user_input: str) -> str:
        # 从向量数据库中查询数据
        similar_reports = self.vector_memory.find_similar_queries(
            user_input,
            threshold=self.relevance_threshold,
            limit=3
        )
        # 获取对话上下文
        conversation_context = self._get_conversation_context()
        # 获取prompt
        prompt = self._prepare_conversation_prompt(
            user_input,
            conversation_context,
            similar_reports
        )
        # 调用LLM生成回复
        try:
            response = self.llm.generate(prompt, temperature=0.7)
            return response.strip()
        except Exception as e:
            return f"生成回应时出错: {str(e)}"

    def _get_conversation_context(self) -> str:
        if not self.conversation_history:
            return ""
        recent_history = self.conversation_history[-2 * self.context_window:]
        context_parts = []
        for message in recent_history:
            role = "用户" if message['role'] == 'user' else "系统"
            content = message['content']
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)

    def _prepare_conversation_prompt(
            self,
            user_input: str,
            conversation_context: str,
            similar_reports: List[Dict]
    ) -> str:
        prompt_parts = [
            "你是一个智能助手，能够基于历史研究报告和可用工具与用户进行多轮对话。",
            "请根据用户的当前问题、对话历史和相关历史研究报告，提供有用的回应。",
        ]

        if conversation_context:
            prompt_parts.append("\n最近的对话历史:")
            prompt_parts.append(conversation_context)

        if similar_reports:
            prompt_parts.append("\n相关历史研究报告:")
            for i, report in enumerate(similar_reports, 1):
                prompt_parts.append(f"{i}. 查询: {report['query']}")
                prompt_parts.append(f"   结果摘要: {report['results_summary']}")
                prompt_parts.append(f"   相似度: {report['similarity']:.2f}")

        prompt_parts.append(f"\n当前用户问题: {user_input}")

        prompt_parts.append(
            "\n请基于以上信息提供有用的回应。如果历史报告中包含相关信息，请引用。"
            "如果需要最新信息，可以提及可以使用搜索工具获取最新数据。"
        )

        return "\n".join(prompt_parts)

