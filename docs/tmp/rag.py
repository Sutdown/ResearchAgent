# 代码示例：RAG ChatBot 完整实现（结构）

print("=== RAG ChatBot 完整项目 ===\n")

# 项目结构
print("1. 项目结构:")
print("""
rag-chatbot/
├── .env                    # 环境变量
├── requirements.txt        # 依赖
├── app.py                  # Streamlit 应用
├── rag/
│   ├── __init__.py
│   ├── embeddings.py      # 嵌入生成
│   ├── vector_store.py    # 向量数据库
│   └── retriever.py       # 检索器
├── data/
│   └── documents/         # 知识库文档
└── logs/
    └── app.log            # 日志
""")

# 核心代码
print("\n2. 核心实现:")
print("""
# embeddings.py
from openai import OpenAI

class EmbeddingGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding


# vector_store.py
import chromadb

class VectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.Client(...)
        self.collection = self.client.get_or_create_collection("docs")

    def add_documents(self, documents, metadatas=None):
        self.collection.add(documents=documents, metadatas=metadatas)

    def query(self, query_text, n_results=3):
        return self.collection.query(query_texts=[query_text], n_results=n_results)


# retriever.py
class RAGRetriever:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def retrieve_and_generate(self, query):
        # 1. 检索相关文档
        results = self.vector_store.query(query, n_results=3)
        context = "\\n".join(results['documents'][0])

        # 2. 构建提示词
        prompt = f'''
基于以下上下文回答问题：

上下文：
{context}

问题：{query}

回答：'''

        # 3. 生成回答
        response = self.llm.generate(prompt)
        return {
            "answer": response,
            "sources": results['documents'][0]
        }


# app.py (Streamlit 应用)
import streamlit as st
from rag import RAGRetriever, VectorStore, EmbeddingGenerator

st.title("🤖 RAG ChatBot")

# 初始化
if "retriever" not in st.session_state:
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store, llm)
    st.session_state.retriever = retriever

# 聊天界面
if prompt := st.chat_input("输入你的问题..."):
    result = st.session_state.retriever.retrieve_and_generate(prompt)

    st.write(result["answer"])
    with st.expander("📚 参考来源"):
        for source in result["sources"]:
            st.write(f"- {source}")
""")

# 工作流程
print("\n3. RAG 工作流程:")
print("""
  ┌─────────────────┐
  │  用户提问       │
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ 生成查询嵌入    │ ← OpenAI Embeddings
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ 向量相似度搜索  │ ← ChromaDB
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ 检索Top-K文档   │
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ 构建提示词      │ ← Query + Context
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ LLM 生成回答    │ ← OpenAI GPT-4
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ 返回答案+来源   │
  └─────────────────┘
""")

# 技术栈
print("\n4. 完整技术栈:")
print("""
  🔹 UI 框架: Streamlit
  🔹 Agent 框架: LangGraph
  🔹 LLM: OpenAI GPT-4
  🔹 Embeddings: OpenAI text-embedding-ada-002
  🔹 向量数据库: ChromaDB
  🔹 数据分析: Pandas
  🔹 可视化: Matplotlib/Plotly
  🔹 日志: Python logging
  🔹 环境管理: python-dotenv
  🔹 部署: Docker + Streamlit Cloud
""")