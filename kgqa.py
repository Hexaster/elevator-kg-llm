import json
import torch
import os
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from typing import List, Dict
from utils import auto_read,eval,embedding_eval,bertscore_eval
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

file_lock = threading.Lock()

class GraphRAGSystem:
    
    def __init__(self, kg_path, embedding_model_path, cache_dir="./kg_cache", llm_api_key="EMPTY", llm_base_url="http://localhost:8000/v1"):
        """
        :param cache_dir: 缓存文件夹路径，用于存储计算好的向量
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # 1. 加载 Embedding 模型
        print(f"🔄 正在加载 Embedding 模型: {embedding_model_path} ...")
        self.embed_model = SentenceTransformer(embedding_model_path, trust_remote_code=True)
        self.embed_model.to(self.device)
        
        # 2. 初始化 LLM 客户端
        self.client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        
        # 3. 加载图谱结构 (每次都需要加载，建立倒排索引)
        self.triples = []
        self.entity_to_triples = defaultdict(list)
        self.unique_entities = [] 
        self.entity_embeddings = None
        
        # 加载数据并处理向量缓存
        self._load_kg_and_embeddings(kg_path)

    def _load_kg_and_embeddings(self, kg_path):
        """加载 JSONL 并尝试读取缓存的 Embedding"""
        print("📖 正在读取图谱文件...")
        entities_set = set()
        
        # 1. 读取 JSONL 构建图结构
        with open(kg_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    triple = json.loads(line)
                    self.triples.append(triple)
                    
                    head = triple['head']
                    tail = triple['tail']
                    
                    self.entity_to_triples[head].append(triple)
                    self.entity_to_triples[tail].append(triple)
                    
                    entities_set.add(head)
                    entities_set.add(tail)
                except:
                    continue
        
        # 关键：将集合转换为排序后的列表，保证顺序一致性
        self.unique_entities = sorted(list(entities_set))
        print(f"✅ 图谱结构加载完成，共 {len(self.unique_entities)} 个唯一实体。")

        # 2. 尝试加载缓存
        cache_emb_path = os.path.join(self.cache_dir, "entity_embeddings.pt")
        cache_ent_path = os.path.join(self.cache_dir, "entities_list.json")
        
        if self._check_cache_valid(cache_ent_path, cache_emb_path):
            print("🚀 检测到有效的本地缓存，正在加载...")
            # 加载 Tensor
            self.entity_embeddings = torch.load(cache_emb_path, map_location=self.device)
            print("✅ 缓存加载成功！")
        else:
            print("⚡ 未检测到缓存或图谱已更新，正在重新计算 Embedding...")
            # 计算向量
            self.entity_embeddings = self.embed_model.encode(
                self.unique_entities, 
                convert_to_tensor=True, 
                show_progress_bar=True
            )
            
            # 保存缓存
            print("💾 正在保存缓存到本地...")
            torch.save(self.entity_embeddings, cache_emb_path)
            with open(cache_ent_path, 'w', encoding='utf-8') as f:
                json.dump(self.unique_entities, f, ensure_ascii=False)
            print("✅ 缓存已保存。")

    def _check_cache_valid(self, ent_path, emb_path):
        """检查缓存是否存在，且内容是否与当前图谱一致"""
        if not os.path.exists(ent_path) or not os.path.exists(emb_path):
            return False
            
        try:
            # 读取缓存的实体列表
            with open(ent_path, 'r', encoding='utf-8') as f:
                cached_entities = json.load(f)
            
            # 简单校验：如果缓存的实体列表和当前图谱解析出的列表完全一致，则认为缓存有效
            # 注意：这里我们使用了 sorted() 保证顺序，可以直接比较列表
            if cached_entities == self.unique_entities:
                return True
            else:
                print("⚠️ 警告：图谱内容发生变化，缓存失效。")
                return False
        except Exception as e:
            print(f"⚠️ 缓存校验出错: {e}")
            return False

    # ... 以下是保持不变的 RAG 逻辑 ...

    def extract_query_entities(self, query: str) -> List[str]:
        # (保持原代码不变)
        system_prompt = """你是一位资深的电梯维修专家。你的核心任务是从用户的电梯维修文本中提取关键实体。
### 请从文本中精准提取以下四类实体：
- **故障现象 (FaultPhenomenon)**: 故障代码（如E30）、错误显示，或异响、震动、运行异常等具体表现。
- **故障原因 (FaultCause)**: 导致故障的技术根源，如“触点氧化”、“电压过低”、“异物卡阻”、“参数设置错误”。
- **组件 (Component)**: 具体的电梯硬件或软件模块，如“门机”、“主板”、“旋转编码器”、“安全回路”。
- **解决方法 (Solution)**: 维修、检查或更换的具体操作，如“更换”、“清洁”、“调整参数”、“紧固螺丝”。

### 输出示例 (One-Shot Example)
请直接返回一个 JSON 字符串列表，不要包含 Markdown 标记，不要包含实体的类型，只返回实体名称即可。
**输入文本：**
"用户投诉电梯停止不能运行, 楼层显示是正常的，但是打不开门"

**输出 JSON：**
[电梯停止, 楼层显示正常, 不开门]
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-r1", 
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            
            content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except:
            return [query]

    def retrieve_related_triples(self, query_entities: List[str], top_k=3, threshold=0.6) -> List[Dict]:
        # (保持原代码不变)
        if not query_entities: return []
        query_embeddings = self.embed_model.encode(query_entities, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embeddings, self.entity_embeddings)
        
        matched_kg_entities = set()
        for i, _ in enumerate(query_entities):
            best_scores, best_indices = torch.topk(cosine_scores[i], k=top_k)
            for score, idx in zip(best_scores, best_indices):
                if score > threshold:
                    matched_kg_entities.add(self.unique_entities[idx])

        retrieved_triples = []
        seen = set()
        for entity in matched_kg_entities:
            for triple in self.entity_to_triples[entity]:
                t_sig = (triple['head'], triple['relation'], triple['tail'])
                if t_sig not in seen:
                    retrieved_triples.append(triple)
                    seen.add(t_sig)
        return retrieved_triples

    def format_triples_to_text(self, triples: List[Dict]) -> str:
        # (保持原代码不变)
        relation_map = {"POSSIBLE_CAUSE": "的可能原因是", "RELATED_TO": "涉及组件", "HAS_SOLUTION": "解决方法是","TARGET_COMPONENT":"目标组件是"}
        return "。".join([f"{t['head']} {relation_map.get(t['relation'], t['relation'])} {t['tail']}" for t in triples])

    def answer_question(self, record: str):
        # (保持原代码不变)
        query = record['fault_description']+record['question']
        # print(f"\n❓ 用户提问: {query}")
        entities = self.extract_query_entities(query)
        # print(entities)
        triples = self.retrieve_related_triples(entities, top_k=5)
        context = self.format_triples_to_text(triples)
        # print(context)
        with open("answer_prompt.txt", 'r', encoding='utf-8') as f:
            user_prompt = f.read()
            
        USER_PROMPT = user_prompt.replace("{{fault_description}}", record['fault_description'])
        USER_PROMPT = USER_PROMPT.replace("{{question}}", record['question'])
        USER_PROMPT = USER_PROMPT.replace("{{reference_content}}", context)
        # print(USER_PROMPT)
        # final_prompt = f"【参考知识】\n{context}\n\n【用户问题】\n{query}"
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": "你是一个电梯维修专家。基于参考知识回答问题。"},
                {"role": "user", "content": USER_PROMPT}
            ]
        )

        return response.choices[0].message.content


# ================= 核心修改部分：多线程处理函数 =================

def process_single_record(record, rag_system, output_file):
    """
    单个记录的处理逻辑：
    1. 调用 RAG 生成
    2. 加锁写入文件
    """
    try:
        # 执行 RAG 逻辑 (耗时操作)
        res = rag_system.answer_question(record)
        record["output"] = res
        
        # 写入文件 (需要加锁，防止多线程同时写乱)
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error processing record: {e}")
        return False

def main(output_file):
    KG_FILE = "triples_aligned.jsonl"
    EMBEDDING_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
    CACHE_DIR = "./my_kg_cache"
    
    # 1. 初始化 RAG 系统 (只初始化一次！共享内存)
    rag = GraphRAGSystem(
        KG_FILE, EMBEDDING_PATH, cache_dir=CACHE_DIR,
        llm_api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    
    records = auto_read("fault_qa.jsonl")
    
    
    # 检查已处理的记录，支持断点续传 (可选优化)
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 假设 record 每一行内容能唯一标识，或者最好有一个 'id' 字段
                    # 这里粗略使用问题内容做去重，建议根据实际情况修改
                    key = data.get('fault_description', '') + data.get('question', '')
                    processed_ids.add(key)
                except: pass
    
    tasks = []
    # 过滤掉已经处理过的
    todo_records = [r for r in records if (r.get('fault_description', '') + r.get('question', '')) not in processed_ids]
    
    print(f"总任务数: {len(records)}, 剩余任务数: {len(todo_records)}")

    # 2. 线程池配置
    # max_workers: 建议设置为 5-10，取决于 API 的并发限制 (Rate Limit)
    # 如果设置太大，API 可能会报 429 Too Many Requests
    MAX_WORKERS = 5
    
    print(f"🚀 开始多线程评测 (Workers: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_record = {
            executor.submit(process_single_record, record, rag, output_file): record 
            for record in todo_records
        }
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_record), total=len(todo_records)):
            try:
                future.result() # 获取结果，如果有异常会在这里抛出
            except Exception as exc:
                print(f'Task generated an exception: {exc}')
    
# ================= 主程序 =================

if __name__ == "__main__":
    output_file = "fault_qa_deepseek-r1_kgrag_multi.jsonl"
    # main(output_file)
    # eval(output_file)
    bertscore_eval("./data/output/fault_qa_deepseek-r1.jsonl")
    bertscore_eval("./data/fault_qa_deepseek-r1_kgrag_multi.jsonl")
    
    # bertscore_eval("./data/output/fault_qa_deepseek-v3.jsonl")
    # bertscore_eval("./data/fault_qa_deepseek-v3_kgrag_multi.jsonl")
    
    # bertscore_eval("./data/output/fault_qa_o1.jsonl")
    # bertscore_eval("./data/fault_qa_o1_kgrag_multi.jsonl")
    
    # bertscore_eval("./data/output/fault_qa_gpt-4o.jsonl")
    # bertscore_eval("./data/fault_qa_4o_kgrag_multi.jsonl")
    
    # embedding_eval("1.jsonl")
    print("✅ 所有评测任务完成！")
# # ================= 使用示例 =================
# if __name__ == "__main__":
#     KG_FILE = "triples_aligned.jsonl"
#     EMBEDDING_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B") # 修改路径
#     CACHE_DIR = "./my_kg_cache" # 缓存存放位置
    
#     # 第一次运行：会计算并保存到 ./my_kg_cache
#     rag = GraphRAGSystem(KG_FILE, EMBEDDING_PATH, cache_dir=CACHE_DIR,
#                          llm_api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
#                         llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
#                          )
    
#     records = auto_read("fault_qa.jsonl")
#     output_file = "fault_qa_4o_kgrag.jsonl"
#     for record in tqdm(records):
#         res = rag.answer_question(record)
#         with open(output_file, 'a', encoding='utf-8') as f:
#                 record["output"] = res
                
#                 f.write(json.dumps(record, ensure_ascii=False) + '\n')
                

#     # 第二次运行（重启程序后）：会直接从 ./my_kg_cache 加载，速度飞快