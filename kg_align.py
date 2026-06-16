import json
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os

class EntityAligner:
    def __init__(self, model_path, threshold=0.95, device='cuda'):
        """
        :param model_path: 本地 Qwen-embedding-0.6B 模型路径
        :param threshold: 相似度阈值 (0.95)
        :param device: 运行设备 ('cuda' for 3090)
        """
        self.threshold = threshold
        self.device = device
        
        print(f"🔄 正在加载模型: {model_path} ...")
        # Qwen-embedding 通常兼容 SentenceTransformer
        # 如果是 HuggingFace 格式，SentenceTransformer 可以直接加载
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.model.to(device)
        print("✅ 模型加载完成")

    def align_and_merge(self, input_file, output_file):
        # 1. 读取数据并提取所有实体
        triples = []
        entities_by_type = {} # 格式: {'FaultPhenomenon': {'电梯停了', '电梯停车', ...}}

        print("📖 读取三元组数据...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                triples.append(data)
                
                # 收集实体 (按类型分组)
                self._add_entity(entities_by_type, data['head'], data['type'])
                self._add_entity(entities_by_type, data['tail'], data['tail_type'])

        # 2. 生成映射字典 (旧名字 -> 新名字)
        global_mapping = {}
        
        print(f"🔍 开始实体对齐 (阈值: {self.threshold})...")
        for etype, names_set in entities_by_type.items():
            names_list = list(names_set)
            if len(names_list) < 2:
                continue
                
            print(f"   正在处理类型 [{etype}]: {len(names_list)} 个实体")
            
            # 计算 embedding
            embeddings = self.model.encode(names_list, convert_to_tensor=True, show_progress_bar=False)
            
            # 计算相似度矩阵 (Cosine Similarity)
            # embeddings 已经在 GPU 上，计算非常快
            cos_scores = util.cos_sim(embeddings, embeddings)
            
            # 构建图来处理传递性 (A~B, B~C -> A,B,C 是一组)
            G = nx.Graph()
            G.add_nodes_from(names_list)
            
            # 找到相似度 > 0.95 的对，添加边
            # 只需要遍历上三角矩阵 (i < j)
            rows, cols = torch.where(cos_scores > self.threshold)
            
            for i, j in zip(rows, cols):
                if i < j: # 排除自身和重复对
                    name_i = names_list[i]
                    name_j = names_list[j]
                    # 添加边
                    G.add_edge(name_i, name_j)
            
            # 找出连通分量 (即聚类结果)
            clusters = list(nx.connected_components(G))
            
            # 对每个聚类选择代表实体
            for cluster in clusters:
                if len(cluster) > 1:
                    # 策略：选择字数最多的作为标准名
                    # 如果字数一样，按字典序选第一个，保证确定性
                    canonical_name = max(cluster, key=lambda x: (len(x), x))
                    
                    # 将簇中其他名字映射到这个标准名
                    for name in cluster:
                        if name != canonical_name:
                            global_mapping[name] = canonical_name
                            # 调试打印
                            # print(f"      🔗 合并: '{name}' -> '{canonical_name}'")

        print(f"✅ 对齐完成，共合并了 {len(global_mapping)} 个实体变体。")

        # 3. 替换实体并去重
        print("🔄 正在替换实体并执行最终去重...")
        
        seen_hashes = set()
        final_triples = []
        
        for t in tqdm(triples):
            # 替换 Head
            if t['head'] in global_mapping:
                t['head'] = global_mapping[t['head']]
            
            # 替换 Tail
            if t['tail'] in global_mapping:
                t['tail'] = global_mapping[t['tail']]
            
            # 防止自环 (Head == Tail)，如果合并后导致头尾一样，通常应该丢弃或保留为属性
            if t['head'] == t['tail']:
                continue

            # 构造去重签名
            signature = (t['head'], t['type'], t['relation'], t['tail'], t['tail_type'])
            
            if signature not in seen_hashes:
                seen_hashes.add(signature)
                final_triples.append(t)
        
        # 4. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for t in final_triples:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
                
        print(f"💾 结果已保存至: {output_file}")
        print(f"   原始三元组数: {len(triples)}")
        print(f"   对齐去重后数: {len(final_triples)}")
        

    def _add_entity(self, collection, name, etype):
        if etype not in collection:
            collection[etype] = set()
        collection[etype].add(name)

# ================= 运行脚本 =================
if __name__ == "__main__":
    # 配置
    # 请修改为你的本地模型路径
    # 如果是 HuggingFace 下载的文件夹，直接填文件夹路径
    LOCAL_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B" 
    
    # 也可以填 'Alibaba-NLP/gte-Qwen1.5-7B-instruct' 等如果本地能联网
    # 但你提到有本地 checkpoint，填绝对路径即可
    
    INPUT_FILE = "triples_deduplicated.jsonl" # 上一步的输出
    OUTPUT_FILE = "triples_aligned.jsonl"     # 最终结果
    
   
    # 实例化并运行
    # 确保你有显卡，device='cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 运行 (速度较慢)")
        device = 'cpu'

    aligner = EntityAligner(
        model_path=LOCAL_MODEL_PATH, 
        threshold=0.95, 
        device=device
    )
    
    aligner.align_and_merge(INPUT_FILE, OUTPUT_FILE)