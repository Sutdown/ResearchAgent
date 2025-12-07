import unittest
import os
from RAgents.tools.tavily_search import TavilySearch

class TestTavilySearch(unittest.TestCase):

    def setUp(self):
        self.api_key = "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"
        self.tavily_search = TavilySearch(api_key=self.api_key)

    def test_search(self):
        query = "Python 单元测试"
        result = self.tavily_search.search(query=query, max_results=2)
        print(result)
        self.assertIn('results', result)
        self.assertIsInstance(result['results'], list)

    def test_get_search_context(self):
        query = "Python 单元测试"
        context = self.tavily_search.get_search_context(query=query, max_results=2, max_chars=500)
        print(context)
        self.assertIsInstance(context, str)
        self.assertTrue(len(context) > 0)

if __name__ == "__main__":
    unittest.main()
