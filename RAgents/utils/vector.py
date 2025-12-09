import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

try:
    import chromadb # chromadb 是用于向量存储的库
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer # 用于嵌入文本的模型
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

class VectorMemory:
    def __init__(self, persist_directory: str = "./vector_memory", embedding_model: str = "all-MiniLM-L6-v2"):
        # 设置存储目录和嵌入模型
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        # 初始化嵌入模型
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model) # 使用 SentenceTransformer 模型
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension() # 获取嵌入维度
        else:
            print("Warning: Using fallback simple text matching (no semantic search)")
            self.embedding_model = None
            self.embedding_dim = 0
        # 初始化向量库
        if CHROMADB_AVAILABLE and self.embedding_model:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="research_memory",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = None
            self.fallback_storage = {}
        # 初始化缓存
        self.recent_cache = {}    # 最近访问的缓存
        self.cache_max_size = 100 # 缓存最大大小
        self.cache_expiry_hours = 2 # 缓存过期时间（小时）

    # 存储新的研究成果
    def store_research_result(self, query: str, results: Dict, quality_score: float = 0.0, metadata: Dict = None):
        try:
            query_id = self._generate_query_id(query)
            document = {
                'query': query, # 查询问题
                'results_summary': self._summarize_results(results), # 摘要
                'quality_score': quality_score,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'query_id': query_id
            }
            self._update_cache(query_id, document) # 更新新的数据

            # 存储到chromaDB
            if self.collection and self.embedding_model:
                # 使用嵌入模型计算嵌入向量，便于之后的索引查询
                query_embedding = self.embedding_model.encode(query).tolist()

                self.collection.add(
                    ids=[query_id],
                    embeddings=[query_embedding],
                    documents=[json.dumps(document)],
                    metadatas={
                        'query': query,
                        'quality_score': str(quality_score),
                        'timestamp': document['timestamp']
                    }
                )
            else:
                self.fallback_storage[query_id] = document
        except Exception as e:
            print(f"Error storing research result: {e}")

    # 查找相似的研究成果
    def find_similar_queries(self, query: str, threshold: float = 0.8, limit: int = 5) -> List[Dict]:
        try:
            # 先在缓存中寻找
            cached_results = self._check_cache(query)
            if cached_results:
                return cached_results
            # 然后在chromaDB中寻找
            if self.collection and self.embedding_model:
                query_embedding = self.embedding_model.encode(query).tolist()
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit
                )

                similar_queries = []
                for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    # 余弦距离转化成相似度，加载到达阈值的结果
                    similarity = 1 - distance
                    if similarity >= threshold:
                        document = json.loads(results['documents'][0][i])
                        similar_queries.append({
                            'query': document['query'],
                            'results_summary': document['results_summary'],
                            'similarity': similarity,
                            'quality_score': document['quality_score'],
                            'timestamp': document['timestamp'],
                            'query_id': document['query_id']
                        })
                return similar_queries
            return self._fallback_similarity_search(query, limit)
        except Exception as e:
            print(f"Error finding similar queries: {e}")
            return []

    # 更新质量分数
    def update_quality_score(self, query_id: str, new_score: float):
        try:
            if self.collection:
                # 修改document
                results = self.collection.get(ids=[query_id])
                if results['ids']:
                    # 更新documents
                    document = json.loads(results['documents'][0])
                    document['quality_score'] = new_score
                    document['updated_timestamp'] = datetime.now().isoformat()

                    self.collection.update(
                        ids=[query_id],
                        documents=[json.dumps(document)],
                        metadatas={
                            'query': document['query'],
                            'quality_score': str(new_score), # 更新metadatas
                            'updated_timestamp': document['updated_timestamp']
                        }
                    )
        except Exception as e:
            print(f"Error updating quality score: {e}")

    # 组件
    def _generate_query_id(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()[:16]

    def _summarize_results(self, results: Dict) -> str:
        if isinstance(results, str):
            return results[:500] + "..." if len(results) > 500 else results
        elif isinstance(results, dict):
            summary_parts = []
            if 'search_results' in results:
                summary_parts.append(f"Found {len(results['search_results'])} results")
            if 'final_report' in results:
                summary_parts.append("Report generated")
            return " | ".join(summary_parts)
        else:
            return str(results)[:500]

    def _update_cache(self, query_id: str, document: Dict):
        # 检查缓存大小，满时删除最远的时间戳的报告
        if len(self.recent_cache) >= self.cache_max_size:
            oldest_key = min(self.recent_cache.keys(),
                             key=lambda k: self.recent_cache[k]['timestamp'])
            del self.recent_cache[oldest_key]
        # 添加新的报告
        self.recent_cache[query_id] = {
            'data': document,
            'timestamp': datetime.now()
        }

    def _check_cache(self, query: str) -> Optional[List[Dict]]:
        # 检查缓存，在最近缓存中找到最相似的三条
        query_words = set(query.lower().split())
        query_lower = query.lower()

        similar_results = []
        for cached_id, cached_item in self.recent_cache.items():
            if self._is_cache_expired(cached_item):
                continue

            cached_doc = cached_item['data']
            cached_words = set(cached_doc['query'].lower().split())
            cached_lower = cached_doc['query'].lower()

            # 1. 词汇重叠度 (30%)，最简单快速
            overlap = len(query_words & cached_words)
            word_sim = overlap / max(len(query_words), len(cached_words))

            # 2. 字符串相似度 (40%)，考虑词汇的顺序和上下文关系，同时考虑到了同义词
            from difflib import SequenceMatcher
            str_sim = SequenceMatcher(None, query_lower, cached_lower).ratio()

            # 3. 编辑距离相似度 (20%)，两个词之间的变化程序
            import Levenshtein  # 需要安装 python-Levenshtein
            edit_dist = Levenshtein.distance(query_lower, cached_lower)
            max_len = max(len(query_lower), len(cached_lower))
            edit_sim = 1 - (edit_dist / max_len) if max_len > 0 else 0

            # 4. 长度相似度 (10%)
            len_sim = 1 - abs(len(query) - len(cached_doc['query'])) / max(len(query), len(cached_doc['query']))

            # 加权平均
            combined_sim = (
                    word_sim * 0.3 +
                    str_sim * 0.4 +
                    edit_sim * 0.2 +
                    len_sim * 0.1
            )

            if combined_sim > 0.55:  # 稍微降低阈值
                similar_results.append({
                    'query': cached_doc['query'],
                    'results_summary': cached_doc['results_summary'],
                    'similarity': combined_sim,
                    'quality_score': cached_doc['quality_score'],
                    'timestamp': cached_doc['timestamp'],
                    'query_id': cached_id
                })

        return sorted(similar_results, key=lambda x: x['similarity'], reverse=True)[:3]

    def _is_cache_expired(self, cached_item: Dict) -> bool:
        # 检查是否过期
        age = datetime.now() - cached_item['timestamp']
        return age > timedelta(hours=self.cache_expiry_hours)

    # 降级策略
    def _fallback_similarity_search(self, query: str, limit: int) -> List[Dict]:
        query_words = set(query.lower().split())
        similar_results = []

        for stored_id, stored_doc in self.fallback_storage.items():
            stored_words = set(stored_doc['query'].lower().split())
            # Calculate Jaccard similarity
            intersection = len(query_words & stored_words)
            union = len(query_words | stored_words)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0.3:  # Lower threshold for fallback
                similar_results.append({
                    'query': stored_doc['query'],
                    'results_summary': stored_doc['results_summary'],
                    'similarity': similarity,
                    'quality_score': stored_doc['quality_score'],
                    'timestamp': stored_doc['timestamp'],
                    'query_id': stored_id
                })

        return sorted(similar_results, key=lambda x: x['similarity'], reverse=True)[:limit]

