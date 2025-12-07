from RAgents.tools.arxiv_search import ArxivSearch


def test_arxiv_search():
    arxiv_client = ArxivSearch()

    print("=== 1. 测试关键词搜索 ===")
    query = "large language model"
    search_result = arxiv_client.search(query=query, max_results=2)
    print(f"查询：{query}")
    print(f"返回论文数：{search_result['total_results']}")
    for idx, paper in enumerate(search_result["results"], start=1):
        print(f"{idx}. {paper['title']}")

    if not search_result["results"]:
        print("搜索失败，无论文返回")
        return

    print("\n=== 2. 使用第一篇论文的 ID 测试 get_paper_by_id ===")
    first_paper_url = search_result["results"][0]["url"]
    paper_id = first_paper_url.split("/")[-1]

    paper_info = arxiv_client.get_paper_by_id(paper_id)
    print(f"查询 ID：{paper_id}")
    print("论文标题：", paper_info["title"])
    print("作者：", paper_info["authors"])

    print("\n=== 3. 测试下载 PDF（只下载第一篇）===")
    pdf_path = arxiv_client.download_pdf(paper_id, dirpath="./tests/")
    print("PDF 本地路径：", pdf_path)


if __name__ == "__main__":
    test_arxiv_search()
