from typing import List, Dict, Optional
from tavily import TavilyClient
from datetime import datetime

class TavilySearch:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key)

    def search(
            self,
            query: str, # 问题
            max_results: int = 3,        # 返回的网页条目数量
            search_depth: str = "basic", # 搜索深度，basic（默认），advanced（信息更丰富）
            include_domains: Optional[List[str]] = None, # 限制包含的域名
            exclude_domains: Optional[List[str]] = None, # 限制排除的域名
    ) -> Dict:
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),  # 标题
                    'url': item.get('url', ''),      # 链接
                    'snippet': item.get('content', ''),        # 摘要
                    'relevance_score': item.get('score', 0.0), # 相关性分数
                    'metadata': {
                        'published_date': item.get('published_date'), # 发布日期
                        'raw_content': item.get('raw_content')        # 原始内容
                    }
                })

            return {
                'query': query,
                'source': 'tavily',
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results)
            }

        except Exception as e:
            return {
                'query': query,
                'source': 'tavily',
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_search_context(
            self,
            query: str,
            max_results: int = 5,
            max_chars: int = 5000,  # 限制 context 长度，防止 prompt 太长
    ) -> str:
        try:
            result = self.client.search(
                query=query,
                max_results=max_results,
            )

            items = []
            for r in result.get("results", []):
                items.append({
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })

            import json
            context = json.dumps(items, ensure_ascii=False, indent=2)

            # 超过长度就裁切
            if len(context) > max_chars:
                context = context[:max_chars] + "\n...<truncated>"

            return context

        except Exception as e:
            return f"Error retrieving search context: {str(e)}"

