import json
from openai import OpenAI
import re



from collections import defaultdict
import json
import asyncio
import os
import hashlib
from typing import List, Dict, Any, Tuple
# 引入 tqdm 的异步版本
from tqdm.asyncio import tqdm as async_tqdm 
from openai import OpenAI
from utils import auto_read
# --- 辅助函数和全局状态 ---

# 确保客户端只初始化一次
_client = None

def get_client(model):
    """线程安全的获取智谱 AI 客户端的函数。"""
    global _client
    if _client is None:
        try:
            _client = OpenAI(
                api_key = os.environ.get("LLM_API_KEY", "EMPTY"),
                base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            )     
        except Exception as e:
            raise RuntimeError(f"客户端初始化失败: {e}")
    return _client

def sync_chat_completion(model: str, messages: List[Dict[str, str]]) -> str:
    client = get_client(model)
    try:
            response = client.chat.completions.create(
            model=model,
            messages=messages,
            # enable_thinking = False,
            temperature=0.1 
        )
    except Exception as e:
        print({e})
    return response


async def aservice(model: str, messages: List[Dict[str, str]]):
    
    response_content = await asyncio.to_thread(
        sync_chat_completion, 
        model, 
        messages
        
    )
    
    return response_content


async def extract_triples_with_llm(chunk: Dict[str, Any], model_name: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    
    text_chunk = chunk['content']
    with open("system_prompt.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()
     
    user_prompt = f"请分析以下文本并提取三元组：\n\n{text_chunk}"

    

    messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
    
    try:
        # **关键修改：使用 async with 包装 API 调用**
        async with semaphore:
            
            response = await aservice(model_name, messages)
        
        

    
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "")
            result = json.loads(content)
        
            # 兼容不同的 JSON 返回结构，确保返回列表
            if isinstance(result, dict) and 'triples' in result:
                return result['triples']
            elif isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # 有些模型可能会把 list 包在一个 key 里，需要根据实际情况调整
                return list(result.values())[0]
        
    except Exception as e:
        print(f"Error extracting: {e}")
        return []

async def classify_and_label_questions(input_file, output_file, model_name, max_concurrency):
    
    
    all_records = []
    
    print(f"正在读取 INPUT 文件: {input_file}...")
    try:
        all_records = auto_read(input_file)
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 包含非法的 JSON 数据。")
        return

    # if os.path.exists(output_file):
    #     print(f"正在读取 OUTPUT 文件 {output_file}，筛选已处理数据...")
    #     with open(output_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             try:
    #                 record = json.loads(line)
                   
    #                 if not record.get('label') :
    #                     processed_ids.add(record['id'])
    #             except json.JSONDecodeError:
    #                 continue # 忽略损坏的行

    # records_to_process = [
    #     record for record in all_records if record['id'] not in processed_ids
    # ]
    records_to_process = all_records

    total_records = len(all_records)
    processed_count = total_records - len(records_to_process)
    # records_to_process = records_to_process[:5]
    if not records_to_process:
        print(f"所有 {total_records} 条记录均已处理完成，无需继续。")
        return

    print(f"总记录数: {total_records} | 已处理: {processed_count} | 待处理: {len(records_to_process)}")
    print(f"设置最大并发 API 调用数限制为: {max_concurrency}")

    
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [extract_triples_with_llm(record, model_name, semaphore) for record in records_to_process]
    system_info = defaultdict(int)

    print("开始分类，使用增量写入模式...")
    

    with open(output_file, 'a', encoding='utf-8') as f:
        
        for future in async_tqdm.as_completed(tasks, total=len(tasks), desc="分类进度"):
            try:
                processed_record = await future
                
                for triple in processed_record:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n致命错误：任务执行失败: {e}")
                
    print(f"\n全部待处理任务完成！总共处理了 {len(records_to_process)} 条新记录。")

    print(system_info)

if __name__ == "__main__":
    

    INPUT_FILE_NAME = "elevator_chunks.jsonl"
    model = "gpt-4o"
    OUTPUT_FILE_NAME = f"triples_{model}.jsonl"
    asyncio.run(classify_and_label_questions(INPUT_FILE_NAME, OUTPUT_FILE_NAME, model_name=model, max_concurrency=5)) 
    