import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT 
import os 
from openai import OpenAI
import random

from api_keys import *

WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
IGNORE_JINA = os.getenv("IGNORE_JINA", "false").lower() == "true"
# Visit Tool (Using Jina Reader)
JINA_READER_URL_PREFIX = "https://r.jina.ai/"

#JINA_API_KEY = os.getenv("JINA_API_KEY")


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
      },
      "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
      }
    },
    "required": ["url", "goal"]
  }
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        response.append(future.result())
                    except Exception as e:
                        response.append(f"Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
    
    def call_server(self, msgs, max_tries=10):
        # 设置 OpenAI 的 API 密钥和 API 基础 URL 使用 vLLM 的 API 服务器。
        from zai import ZhipuAiClient
        
        client = ZhipuAiClient(api_key=ZHIPUAI_API_KEY)

        for attempt in range(max_tries):
            try:
                # chat_response = client.chat.completions.create(
                #     model='/path/qwen2.5-instruct-72b',
                #     messages=msgs,
                #     stop=["\n<tool_response>", "<tool_response>"],
                #     temperature=0.7
                # )
                
                chat_response = client.chat.completions.create(
                    model="qwen2.5-instruct-72b",
                    messages=msgs,
                    stop=["<tool_response>"],
                    temperature=0.7
                )
                
                
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except:
                if attempt == (max_tries - 1):
                    return ""
                continue

    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        max_retries = 3
        timeout = 10
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."

    def crawl4ai_readpage(self, url: str) -> str:
        from crawl4ai import AsyncWebCrawler
        import asyncio
        async def run_crawl():
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url)
                if result :
                    return result
                return None

        try:
            webpage_content = asyncio.run(run_crawl())
            
            if webpage_content is not None:
                return webpage_content
            else:
                print(f"crawl4ai: No text content extracted from {url}")
                return "[visit] Failed to read page."

        except Exception as e:
            print(f"crawl4ai exception while reading {url}: {e}")
            return "[visit] Failed to read page."
    
    def playwright_readpage(self, url: str) -> str:
        from playwright.async_api import async_playwright, TimeoutError
        import asyncio

        async def run_crawl():
            """异步函数：执行 Playwright 浏览器操作和内容提取"""
            browser = None
            try:
                # 启动 Playwright
                async with async_playwright() as p:
                    # 启动 Chromium 浏览器
                    browser = await p.chromium.launch()
                    page = await browser.new_page()

                    # 访问 URL，等待 DOM 加载完成
                    await page.goto(
                        url, 
                        wait_until="domcontentloaded",
                        timeout=60000 
                    )

                    # --- 关键：正文内容提取逻辑 ---
                    # 1. 尝试使用强大的选择器定位主要内容 (如文章、博客)
                    for selector in ["main", "article", "body"]:
                        locator = page.locator(selector).first
                        
                        # 检查元素是否存在且可见，并尝试提取文本
                        if await locator.is_visible(timeout=500):
                            content = await locator.inner_text()
                            
                            # 确保提取的内容有意义，而不是空的导航栏
                            if len(content.strip()) > 100:
                                return content # 成功提取正文，返回文本

                    # 2. 如果启发式失败，回退到获取整个 Body 的可见文本
                    return await page.locator("body").inner_text()
                    
            except TimeoutError:
                return f"[Playwright Error] 导航超时 (60s)."
            except Exception as e:
                # 捕获其他 Playwright 异常
                return f"[Playwright Exception] {e}"
            finally:
                if browser:
                    await browser.close()
                    
        # --- 同步执行区域 ---
        try:
            # 使用 asyncio.run 运行异步爬取函数
            webpage_content = asyncio.run(run_crawl())
            
            if webpage_content and not webpage_content.startswith("[Playwright"):
                # 如果内容存在且不是错误信息，则返回清理后的文本
                return webpage_content.strip()
            else:
                # 如果是 Playwright 错误信息或提取内容为空
                print(f"Playwright: Failed to extract content for {url}. Result: {webpage_content}")
                return "[visit] Failed to read page."

        except Exception as e:
            # 捕获 asyncio.run 自身的错误
            print(f"asyncio exception while reading {url}: {e}")
            return "[visit] Failed to read page."
        
    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            # Alternate between jina and aidata
            #content = self.jina_readpage(url)
            content = self.crawl4ai_readpage(url)
            sevice = "jina"

            # Check if we got valid content
            print(sevice)
            # print(content)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = self.call_server(messages)

                # 如果网页超长，返回结果是 {\n 这种形式
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = self.call_server(messages)
                    summary_retries -= 1
                # 说明 raw 的长度大于10或者已经retry 超出了 
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        # 尝试 parse json
                        raw = json.loads(raw)
                        break
                    except:
                        raw = self.call_server(messages)
                        parse_retry_times += 1
                # parse 失败
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                # parse 成功
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information
