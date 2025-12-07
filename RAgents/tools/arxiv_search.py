from typing import List, Dict, Optional
import arxiv
from datetime import datetime

class ArxivSearch:
    def __init__(self):
        self.client = arxiv.Client()

    def search(
            self,
            query: str,
            max_results: int = 3,
            sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
            sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
    ) -> Dict:
        try:
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )

            results = []
            for paper in self.client.results(search):
                results.append({
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary,
                    'relevance_score': None,  # arXiv doesn't provide relevance scores
                    'metadata': {
                        'authors': [author.name for author in paper.authors],
                        'published': paper.published.isoformat() if paper.published else None,
                        'updated': paper.updated.isoformat() if paper.updated else None,
                        'categories': paper.categories,
                        'primary_category': paper.primary_category,
                        'pdf_url': paper.pdf_url,
                        'doi': paper.doi,
                        'journal_ref': paper.journal_ref,
                        'comment': paper.comment
                    }
                })
            return {
                'query': query,
                'source': 'arxiv',
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results)
            }

        except Exception as e:
            return {
                'query': query,
                'source': 'arxiv',
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    # 根据 paper_id 创建一个 arXiv 搜索，然后从结果中取出那篇论文的完整 metadata 对象
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
            return {
                'title': paper.title,
                'url': paper.entry_id,
                'summary': paper.summary,
                'authors': [author.name for author in paper.authors],
                'published': paper.published.isoformat() if paper.published else None,
                'pdf_url': paper.pdf_url,
                'categories': paper.categories
            }
        except Exception as e:
            return None

    #
    def download_pdf(self, paper_id: str, dirpath: str = "./") -> Optional[str]:
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
            filepath = paper.download_pdf(dirpath=dirpath)
            return filepath
        except Exception as e:
            return None


