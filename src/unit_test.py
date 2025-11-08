from tool_search import *
from tool_visit import *
import unittest
from api_keys import *

os.environ["ZHIPUAI_API_KEY"] = ZHIPUAI_API_KEY

class TestSearch(unittest.TestCase):
    def test_search(self):
        searcher = Search()
        query = ["What is the capital of France?", "Latest news on AI technology"]
        result1 = searcher.zhipu_search(query[0])
        result2 = searcher.zhipu_search(query[1])
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        print("Test search results:")
        print("Query 1 Results:", result1)
        print("Query 2 Results:", result2)


if __name__ == '__main__':
    unittest.main()
    