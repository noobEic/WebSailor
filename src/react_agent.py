import json
import os
from typing import Dict, List, Optional, Union
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer 
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from sentence_transformers import SentenceTransformer
import torch
import sentence_transformers
import time
MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 40))
MAX_TOKEN_LENGTH = int(os.getenv('MAX_LENGTH', 31 * 1024 - 500))

print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')

def calculate_similarity(previous,current,model,method="emb_model"):
    if method == "emb_model":
        embeddings = model.encode(
            [previous, current], 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        sim_matrix = sentence_transformers.util.cos_sim(embeddings[0], embeddings[1])
        similarity = float(sim_matrix[0][0])
        return similarity


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

    def call_server(self, msgs, max_tries=10):
        # Set OpenAI API key and base URL using vLLM API server
        
        # openai_api_key = OPENAI_API_KEY
        # openai_api_base = OPENAI_API_BASE
        
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_responseF", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                )
                content = chat_response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == (max_tries - 1):
                    print(f"vllm server error {e}")
                    return f"vllm server error"
                continue
        
        return "vllm server empty response"

    def count_string_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(tiktoken.encoding_for_model("gpt-4o").encode(text))
    
    def count_tokens(self, messages, model="gpt-4o"):
        try: 
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        except Exception as e: 
            tokenizer = tiktoken.encoding_for_model(model)
        
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        
        return len(tokenizer.encode(full_prompt))

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        answer = data['item']['answer']
        self.user_prompt = user_prompt
        self.user_prompt = self.user_prompt + question
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.user_prompt}]
        
        timing_records = {
            "llm_calls": [],
            "tool_calls": [],
            "final_answer_generation_time": 0
        }
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0

        previous_thought = ""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            
            input_tokens = self.count_tokens(messages)
            start_llm = time.time()
            content = self.call_server(messages)
            end_llm = time.time()
            llm_duration = end_llm - start_llm
            output_tokens = self.count_string_tokens(content)

            current_thought = content.strip()
            
            if '<tool_call>' in current_thought:
                current_thought = current_thought.split('<tool_call>')[0].strip()
            if '<answer>' in current_thought:
                current_thought = current_thought.split('<answer>')[0].strip()
            if '<think>' in current_thought and '</think>' in current_thought:
                try:
                    current_thought = current_thought.split('<think>')[1].split('</think>')[0].strip()
                except IndexError:
                    pass
            
            similarity = 0.0
            if previous_thought:
                similarity = calculate_similarity(previous_thought,current_thought,model)

            if similarity > 0.9:
                print(f"警告: 思考相似度 {similarity:.4f} 超过阈值。强制重新思考。")
                
                # 构建"重新思考"的提示
                rethink_prompt = (
                    "你之前的思考和上一步过于相似。请换个思路，"
                    "探索其他可能性，或者尝试不同的方法。不要重复。"
                )
                
                # 将提示作为 'user' 消息添加，以便模型在下一轮响应
                messages.append({"role": "user", "content": rethink_prompt})
                
                # 跳过本轮的 (坏的) content 追加和工具调用
                continue

            timing_records["llm_calls"].append({
                "round": round, 
                "duration": llm_duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "previous": previous_thought,
                "current": current_thought,
                "similarity": similarity
            })
            
            previous_thought = current_thought

            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    
                    
                    input_tokens = self.count_tokens(messages)
                    start_tool = time.time()
                    result = self._call_tool(tool_name, tool_args)
                    end_tool = time.time()
                    tool_duration = end_tool - start_tool
                    output_tokens = self.count_string_tokens(result)
                    timing_records["tool_calls"].append({
                        "round": round, 
                        "tool_name": tool_name,
                        "duration": tool_duration,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    })
                    
                    
                    
                    
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                timing_records['final_answer_generation_time'] = llm_duration
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = MAX_TOKEN_LENGTH
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                
                
                
                start_llm_final = time.time()
                content = self.call_server(messages)
                end_llm_final = time.time()
                final_llm_duration = end_llm_final - start_llm_final
                timing_records['final_answer_generation_time'] = final_llm_duration
                
                
                
                
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "rollout_id": data['rollout_id'],
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "timing_records": timing_records
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "timing_records": timing_records
        }
        return result
