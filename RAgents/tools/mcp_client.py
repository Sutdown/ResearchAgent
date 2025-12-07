from typing import List, Dict, Optional, Any
from datetime import datetime
import httpx

class MCPClient:
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    # 使用MCP协议执行搜索操作
    async def search(
            self,
            query: str,
            tool_name: str = "web_search",
            **kwargs
    ) -> Dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/tools/{tool_name}",
                    json={
                        "query": query,
                        **kwargs
                    },
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get('results', []):
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('snippet', item.get('content', '')),
                        'relevance_score': item.get('score'),
                        'metadata': item.get('metadata', {})
                    })

                return {
                    'query': query,
                    'source': 'mcp',
                    'tool': tool_name,
                    'results': results,
                    'timestamp': datetime.now().isoformat(),
                    'total_results': len(results)
                }

        except Exception as e:
            return {
                'query': query,
                'source': 'mcp',
                'tool': tool_name,
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    # 获取MCP服务器上所有可用工具的列表
    async def list_tools(self) -> List[Dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/tools",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json().get('tools', [])
        except Exception as e:
            return []

    # 执行特定的MCP工具，比 search 函数更通用
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/tools/{tool_name}",
                    json=parameters,
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                'error': str(e),
                'tool': tool_name
            }
