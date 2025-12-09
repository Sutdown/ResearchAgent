from typing import Dict, List, Callable, Optional
from datetime import datetime
from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader
from RAgents.workflow.state import ResearchState


class Rapporteur:
    def __init__(self, llm: BaseLLM, stream_callback: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.prompt_loader = PromptLoader()
        self.stream_callback = stream_callback

    def generate_report(self, state: ResearchState) -> ResearchState:
        # 提起基本信息
        query = state['query']
        plan = state.get('research_plan', {})
        results = state.get('research_results', [])
        output_format = state.get('output_format', 'markdown')

        # 生成研究摘要
        summary = self._summarize_findings(query, results)

        # 组织信息结构
        organized_info = self._organize_information(summary, results)

        # 生成格式化报告
        if output_format == 'html':
            report = self._generate_html_report(
                query=query,
                plan=plan,
                summary=summary,
                organized_info=organized_info,
                results=results
            )
        else:
            report = self._generate_markdown_report(
                query=query,
                plan=plan,
                summary=summary,
                organized_info=organized_info,
                results=results
            )

        # 更新状态
        state['final_report'] = report
        state['current_step'] = 'completed'
        return state

    def _summarize_findings(self, query: str, results: List[Dict]) -> str:
        seen_titles = set() # 标题
        all_content = [] # 内容

        for result in results:
            for item in result.get('results', []):
                title = item.get('title', 'No title')
                snippet = item.get('snippet', '')

                if title in seen_titles:
                    continue
                seen_titles.add(title)
                all_content.append(f"- {title}: {snippet[:200]}")

        content_text = '\n'.join(all_content[:30]) # 限制30条
        if not content_text.strip():
            content_text = f"- 已为查询'{query}'收集相关研究资料"

        try: # 生成摘要
            prompt = self.prompt_loader.load(
                'rapporteur_summarize',
                query=query,
                research_findings=content_text
            )
            summary = self.llm.generate(prompt, temperature=0.5, max_tokens=1200)

            if not summary or len(summary.strip()) < 50:
                summary = f"已针对'{query}'进行了研究，收集了相关资料和信息。研究发现涵盖多个相关方面，为深入分析提供了基础。"
        except Exception as e:
            print(f"Warning: Summary generation failed: {e}")
            summary = f"对'{query}'的研究已初步完成，收集了相关的研究资料和文献信息。建议查看详细的研究结果和参考资料部分。"

        return summary

    def _organize_information(self, summary: str, results: List[Dict]) -> Dict:
        prompt = self.prompt_loader.load(
            'rapporteur_organize_info',
            summary=summary
        )
        response = self.llm.generate(prompt, temperature=0.5)
        import json
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                organized = json.loads(json_str)
                return organized
        except json.JSONDecodeError:
            pass

        # 兜底策略
        return {
            'themes': [
                {
                    'name': '核心发现',
                    'key_points': [summary[:500]]
                }
            ]
        }

    def _generate_markdown_report(
            self,
            query: str,
            plan: Dict,
            summary: str,
            organized_info: Dict,
            results: List[Dict]
    ) -> str:
        sections = []
        # 基础信息
        sections.append(f"# 研究报告：{query}\n")
        sections.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sections.append(f"**研究目标：** {plan.get('research_goal', query)}\n")
        sections.append(f"**信息来源数量：** {len(results)}\n")
        # 执行摘要
        sections.append("\n## 执行摘要\n")
        sections.append(summary)
        # 核心发现
        sections.append("\n## 核心发现\n")
        for theme in organized_info.get('themes', []):
            sections.append(f"\n### {theme['name']}\n")
            for point in theme.get('key_points', []):
                sections.append(f"- {point}\n")
        # add 深度分析
        sections.append("\n## 深度分析\n")
        sections.append(self._generate_synthesized_analysis(query, summary, organized_info, results))
        # add 结论
        sections.append("\n## 结论\n")
        sections.append(self._generate_conclusion(query, summary))
        # add 参考文献
        sections.append("\n## 参考文献\n")
        sections.append(self._format_citations(results))
        return '\n'.join(sections)

    def _generate_html_report(
            self,
            query: str,
            plan: Dict,
            summary: str,
            organized_info: Dict,
            results: List[Dict]
    ) -> str:
        analysis = self._generate_synthesized_analysis(query, summary, organized_info, results)
        conclusion = self._generate_conclusion(query, summary)
        citations = self._format_citations(results)
        # 生成主题列表
        themes_text = ""
        for theme in organized_info.get('themes', []):
            themes_text += f"<h3>{theme['name']}</h3>\n<ul>\n"
            for point in theme.get('key_points', []):
                themes_text += f"<li>{point}</li>\n"
            themes_text += "</ul>\n"

        try: # 生成HTML报告
            prompt = self.prompt_loader.load(
                'rapporteur_generate_html',
                query=query,
                research_goal=plan.get('research_goal', query),
                summary=summary[:1000] if summary else "",  # Limit summary size
                themes=themes_text[:1500] if themes_text else "",  # Limit themes size
                analysis=analysis[:1000] if analysis else "",  # Limit analysis size
                citations=citations[:2000] if citations else "",  # Limit citations size
                conclusion=conclusion[:800] if conclusion else ""  # Limit conclusion size
            )
            html_report = self.llm.generate(prompt, temperature=0.3, max_tokens=2000)
            # 去除代码块
            if '```html' in html_report:
                html_report = html_report.split('```html')[1].split('```')[0].strip()
            elif '```' in html_report:
                html_report = html_report.split('```')[1].split('```')[0].strip()

            # 兜底策略
            if not html_report or len(html_report.strip()) < 100:
                html_report = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>研究报告：{query}</title>
                    <meta charset="utf-8">
                </head>
                <body>
                    <h1>研究报告：{query}</h1>
                    <h2>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
                    <p>HTML报告生成遇到问题，但研究内容已正常收集。建议使用Markdown格式查看完整报告。</p>
                    <h2>核心发现</h2>
                    <p>{summary[:500] if summary else '研究已收集相关资料'}</p>
                    <h2>结论</h2>
                    <p>{conclusion[:300] if conclusion else '研究已完成'}</p>
                </body>
                </html>
                """
            return html_report

        except Exception as e:
            print(f"Warning: HTML generation failed: {e}")
            # Fallback HTML
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>研究报告：{query}</title>
                <meta charset="utf-8">
            </head>
            <body>
                <h1>研究报告：{query}</h1>
                <h2>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
                <p>HTML生成过程遇到技术问题，但研究内容已成功收集。</p>
                <h2>核心摘要</h2>
                <p>{summary[:300] if summary else '研究资料收集完成'}</p>
                <h2>分析结论</h2>
                <p>{conclusion[:200] if conclusion else '分析已完成'}</p>
            </body>
            </html>
            """

    def _generate_synthesized_analysis(
            self,
            query: str,
            summary: str,
            organized_info: Dict,
            results: List[Dict]
    ) -> str:
        key_content = []
        for result in results[:10]:  # Limit to first 10 results
            for item in result.get('results', [])[:3]:  # Top 3 per result
                key_content.append(f"- {item.get('snippet', '')[:300]}")
        content_text = '\n'.join(key_content)
        prompt = self.prompt_loader.load(
            'rapporteur_synthesized_analysis',
            query=query,
            summary=summary[:800],
            key_content=content_text
        )

        if self.stream_callback is not None:
            chunks: List[str] = []
            try:
                for chunk in self.llm.stream_generate(prompt, temperature=0.6, max_tokens=1200):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    self.stream_callback(chunk)  # 将增量内容交给上层（例如 CLI）实时输出
                analysis = "".join(chunks)
            except Exception as e:
                print(f"Warning: Stream generation failed, falling back to regular generation: {e}")
                analysis = self.llm.generate(prompt, temperature=0.6, max_tokens=800)
        else:
            try:
                analysis = self.llm.generate(prompt, temperature=0.6, max_tokens=1200)
            except Exception as e:
                print(f"Warning: Analysis generation failed, using shorter fallback: {e}")
                analysis = "深度分析生成遇到问题，但研究已收集了相关的核心信息。"
        if not analysis or len(analysis.strip()) < 50:
            analysis = "基于收集的研究资料，已对相关主题进行了系统性分析。详细信息请参考核心发现和参考资料部分。"
        return analysis

    def _generate_conclusion(self, query: str, summary: str) -> str:
        prompt = self.prompt_loader.load(
            'rapporteur_conclusion',
            query=query,
            summary=summary[:1000] if summary else ""
        )
        try:
            conclusion = self.llm.generate(prompt, temperature=0.5, max_tokens=600)
            if not conclusion or len(conclusion.strip()) < 30:
                conclusion = f"基于对'{query}'的研究，已收集并整理了相关资料。建议用户根据具体需求进一步深入研究特定方面。"
        except Exception as e:
            print(f"Warning: Conclusion generation failed: {e}")
            conclusion = f"本研究对'{query}'进行了系统性调研，收集了相关信息和数据。研究结果可为进一步的深入分析提供基础。"
        return conclusion

    def _format_citations(self, results: List[Dict]) -> str:
        seen_urls = set()
        seen_titles = set()
        citations = []
        citation_num = 1

        for result in results:
            for item in result.get('results', []):
                title = item.get('title', 'Untitled')
                url = item.get('url', '')
                source = result.get('source', 'Unknown')
                # 过滤重复的
                if url and url in seen_urls:
                    continue
                if title and title in seen_titles:
                    continue
                # 记录已见过的
                if url:
                    seen_urls.add(url)
                if title:
                    seen_titles.add(title)
                # 在最终结果加上当前结果
                if url:
                    citations.append(f"{citation_num}. {title} - {source.capitalize()} - [{url}]({url})")
                else:
                    citations.append(f"{citation_num}. {title} - {source.capitalize()}")
                citation_num += 1
        return '\n'.join(citations[:50])


