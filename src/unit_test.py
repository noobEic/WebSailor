from tool_search import *
from tool_visit import *
import unittest
from api_keys import *

os.environ["ZHIPUAI_API_KEY"] = ZHIPUAI_API_KEY

class Test(unittest.TestCase):
    # def test_search(self):
    #     searcher = Search()
    #     query = ["What is the capital of France?", "Latest news on AI technology"]
    #     result1 = searcher.zhipu_search(query[0])
    #     result2 = searcher.zhipu_search(query[1])
    #     self.assertIsNotNone(result1)
    #     self.assertIsNotNone(result2)
        
    #     print("Test search results:")
    #     print("Query 1 Results:", result1)
    #     print("Query 2 Results:", result2)

    def test_visit(self):
        visitor = Visit()
        urls = [
            "http://mp.weixin.qq.com/s?__biz=MzIwNDY1NzAzMQ==&mid=2247497474&idx=1&sn=b7abcf59cafcd3b99f43e174346c357b",
        ]
        for url in urls:
            content = visitor.jina_readpage(url)
            self.assertIsNotNone(content)
            print(content)
            
    

if __name__ == '__main__':
    unittest.main()
    