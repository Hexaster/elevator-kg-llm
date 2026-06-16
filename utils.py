import json
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI

def parse_json_response(text_response):
    """
    解析模型返回的 JSON Schema 格式的文本
    
    Args:
        text_response: 模型返回的文本（JSON 格式字符串）
    
    Returns:
        list: 解析后的数据列表
    """
    try:
        # 去除可能的空白字符和 Markdown 代码块标记
        text_response = text_response.strip()
        
        # 处理可能的代码块标记
        if text_response.startswith('```'):
            # 移除开头的 ```json 或 ``` 标记
            lines = text_response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # 移除结尾的 ``` 标记
            if lines[-1].startswith('```'):
                lines = lines[:-1]
            text_response = '\n'.join(lines)
        
        # 解析 JSON
        data = json.loads(text_response)
        
        # 验证数据结构
        if not isinstance(data, list):
            raise ValueError("返回的数据不是列表格式")
        
        # 验证每个元素的结构
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"第 {i+1} 个元素不是字典格式")
            
            # 检查必需字段
            required_fields = ['fault_description', 'question', 'answer', 'keywords']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"第 {i+1} 个元素缺少字段: {field}")
            
            # 验证字段类型
            if not isinstance(item['fault_description'], str):
                raise ValueError(f"第 {i+1} 个元素的 fault_description 不是字符串")
            if not isinstance(item['question'], str):
                raise ValueError(f"第 {i+1} 个元素的 question 不是字符串")
            if not isinstance(item['answer'], str):
                raise ValueError(f"第 {i+1} 个元素的 answer 不是字符串")
            if not isinstance(item['keywords'], list):
                raise ValueError(f"第 {i+1} 个元素的 keywords 不是列表")
            
            # 验证 keywords 中的每个元素都是字符串
            for j, keyword in enumerate(item['keywords']):
                if not isinstance(keyword, str):
                    raise ValueError(f"第 {i+1} 个元素的 keywords[{j}] 不是字符串")
        
        print(f"成功解析 {len(data)} 条数据")
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print(f"原始文本: {text_response[:200]}...")  # 显示前200个字符
        return []
    except ValueError as e:
        print(f"数据格式错误: {e}")
        return []
    except Exception as e:
        print(f"解析过程中发生未知错误: {e}")
        return []
    
import json
import os
from typing import List, Dict, Any, Union
import logging

def read_json(file_path: str) -> Union[List, Dict, None]:
       
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(e)
        return None
    except Exception as e:
        print(e)
        return None
    
def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    
        
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                obj = json.loads(line)
                data.append(obj)
        return data
        
    except Exception as e:
        print(e)
        return None

def auto_read(file_path: str) -> Union[List, Dict, List[Dict], None]:
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    # 检查文件扩展名
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.json':
        return read_json(file_path)
    elif ext == '.jsonl' or file_path.endswith('.jsonl'):
        return read_jsonl(file_path)
    



import re
from typing import List, Dict, Any
from collections import defaultdict

def clean_text(text: str) -> str:
    """
    清理文本：去除冗余空格，将字母转为小写，去除首尾空格
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def calculate_keyword_recall(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算关键词召回率
    
    参数:
    data_list: 数据列表，每个元素包含 keywords 和 output 字段
    
    返回:
    包含统计结果的字典
    """
    if not data_list:
        return {
            "total_samples": 0,
            "avg_recall": 0,
            "recall_distribution": {},
            "detailed_results": []
        }
    
    total_samples = len(data_list)
    total_recall = 0
    recall_scores = []
    detailed_results = []
    
    # 用于统计召回率分布
    recall_distribution = defaultdict(int)
    
    for i, data in enumerate(data_list):
        # 提取并清理关键词和输出
        keywords = data.get('keywords', [])
        output = data.get('output', '')
        if not keywords:
            # 如果没有关键词，召回率为0
            recall = 0
            matched_keywords = []
            missed_keywords = []
        else:
            # 清理关键词和输出
            cleaned_output = clean_text(output)
            cleaned_keywords = [clean_text(kw) for kw in keywords]
            # 统计匹配的关键词
            matched_keywords = []
            missed_keywords = []
            
            for kw in cleaned_keywords:
                # 精准匹配：关键词作为一个完整单词/短语出现在输出中
                # 使用正则表达式确保精准匹配（考虑中文和英文）
                # pattern = r'(^|\s)' + re.escape(kw) + r'($|\s|[.,;:!?])'
                pattern = re.escape(kw)
                if re.search(pattern, cleaned_output):
                    matched_keywords.append(kw)
                else:
                    missed_keywords.append(kw)
            
            # 计算召回率
            recall = len(matched_keywords) / len(cleaned_keywords) if cleaned_keywords else 0
            
        detailed_result = {
            'sample_id': i + 1,
            'original_keywords': keywords,
            'original_output': output,
            'cleaned_keywords': cleaned_keywords if 'cleaned_keywords' in locals() else [],
            'cleaned_output': cleaned_output if 'cleaned_output' in locals() else '',
            'matched_keywords': matched_keywords,
            'missed_keywords': missed_keywords,
            'recall': round(recall, 4)
        }
        detailed_results.append(detailed_result)
        
        # 更新统计信息
        total_recall += recall
        recall_scores.append(recall)
        
        # 统计召回率分布（按0.1间隔）
        recall_bucket = int(recall * 10) / 10  # 0.0, 0.1, 0.2, ..., 1.0
        recall_distribution[recall_bucket] += 1
    
    # 计算平均召回率
    avg_recall = total_recall / total_samples if total_samples > 0 else 0
    
    # 计算召回率统计
    if recall_scores:
        max_recall = max(recall_scores)
        min_recall = min(recall_scores)
    else:
        max_recall = min_recall = 0
    
    return {
        "total_samples": total_samples,
        "avg_recall": round(avg_recall, 4),
        "max_recall": round(max_recall, 4),
        "min_recall": round(min_recall, 4),
        "recall_distribution": dict(sorted(recall_distribution.items())),
        "perfect_recall_count": sum(1 for r in recall_scores if r == 1.0),
        "zero_recall_count": sum(1 for r in recall_scores if r == 0.0),
        "detailed_results": detailed_results
    }

def print_statistics(statistics: Dict[str, Any], show_details: bool = False):
    """
    打印统计结果
    
    参数:
    statistics: calculate_keyword_recall 返回的统计结果
    show_details: 是否显示每个样本的详细结果
    """
    print("=" * 60)
    print("关键词召回率评估结果")
    print("=" * 60)
    print(f"总样本数: {statistics['total_samples']}")
    print(f"平均召回率: {statistics['avg_recall']:.2%}")
    print(f"最高召回率: {statistics['max_recall']:.2%}")
    print(f"最低召回率: {statistics['min_recall']:.2%}")
    print(f"完美召回样本数: {statistics['perfect_recall_count']}")
    print(f"零召回样本数: {statistics['zero_recall_count']}")
    
    print("\n召回率分布:")
    for bucket, count in statistics['recall_distribution'].items():
        percentage = count / statistics['total_samples'] * 100
        print(f"  {bucket:.1f}-{bucket+0.1:.1f}: {count} 个样本 ({percentage:.1f}%)")
    
    if show_details and statistics['detailed_results']:
        print("\n详细结果:")
        for result in statistics['detailed_results']:
            print(f"\n样本 {result['sample_id']}:")
            print(f"  原始关键词: {result['original_keywords']}")
            print(f"  原始输出: {result['original_output'][:100]}...")  # 只显示前100字符
            print(f"  匹配关键词: {result['matched_keywords']}")
            print(f"  未匹配关键词: {result['missed_keywords']}")
            print(f"  召回率: {result['recall']:.2%}")

def eval(input_file):
    
    sample_data = auto_read(input_file)
    stats = calculate_keyword_recall(sample_data)
    
    # 打印统计结果
    # print_statistics(stats, show_details=True)
    
    # 也可以只获取统计摘要
    print("\n" + "=" * 60)
    print("统计摘要:")
    print(f"平均召回率: {stats['avg_recall']:.2%}")
    print(f"完美匹配的样本数: {stats['perfect_recall_count']}/{stats['total_samples']}")

def embedding_eval(input_file):
    def get_embeddings(texts, model_name="bert-base-chinese"):
        client = OpenAI(
        
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),  
        
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=texts
        )
        embeddings = np.array([item.embedding for item in completion.data])
        return embeddings
    def cosine_similarity( vec1, vec2):
        
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        dot_product = np.dot(vec1, vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    

    data = auto_read(input_file)
 
    total_similarity = 0
    for i in tqdm(range(0,len(data))):
        if 'output' not in data[i] or 'answer' not in data[i]:
            print("样本缺少 'output' 或 'answer' 字段，跳过该样本。")
            continue

        data[i]['output_embeddings']=get_embeddings([data[i]['output']])
        data[i]['answer_embeddings']=get_embeddings([data[i]['answer']])
        sim = cosine_similarity(data[i]['output_embeddings'], data[i]['answer_embeddings'])
        total_similarity += sim

        data[i]['output_embeddings'] = data[i]['output_embeddings'].tolist()
        data[i]['answer_embeddings'] = data[i]['answer_embeddings'].tolist()

   
    

    avg_similarity = total_similarity / len(data) if len(data) > 0 else 0
    print("\n" + "=" * 60)
    print("Embedding 余弦相似度评估结果:")
    print(f"平均余弦相似度: {avg_similarity:.4f}")
    with open(input_file, 'w', encoding='utf-8') as f:
    
        for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

from bert_score import score
from transformers import BertTokenizer
def bertscore_eval(input_file):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def truncate_text(text, max_tokens=500):
        """手动截断文本"""
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens, skip_special_tokens=True)

    data = auto_read(input_file)
 
    total_similarity = 0
    max_sim = 0
    for i in tqdm(range(0,len(data))):
        if 'output' not in data[i] or 'answer' not in data[i]:
            print("样本缺少 'output' 或 'answer' 字段，跳过该样本。")
            continue
        cands = [data[i]['output']]
        refs = [data[i]['answer']]
        # print(cands,refs)
        # print(cands)

        # 处理cands和refs
        cands_truncated = [truncate_text(text) for text in cands]
        refs_truncated = [truncate_text(text) for text in refs]
        P, R, F1 = score(cands_truncated, refs_truncated,model_type="bert-base-chinese",lang="zh")

        total_similarity += F1.mean().item()
        max_sim = max(max_sim,F1.mean().item())

    avg_similarity = total_similarity / len(data) if len(data) > 0 else 0
    print("\n" + "=" * 60)
    print(input_file)
    print("Bertscore结果:")
    print(f"平均相似度: {avg_similarity:.4f}")
    print(f"最高相似度: {max_sim:.4f}")
    
    return avg_similarity