import json
from collections import Counter

def clean_and_analyze_triples(input_path, output_path):
    """
    1. 去除重复的三元组行
    2. 统计实体类型数量（基于唯一实体）
    3. 统计关系类型数量
    """
    
    # 用于去重的集合
    seen_triples = set()
    
    # 用于统计的数据结构
    # relation_counter: 统计边的数量
    relation_counter = Counter()
    
    # unique_entities: 统计唯一的节点 (名称, 类型)
    # 使用集合是为了避免同一个实体在多行中出现导致重复计数
    unique_entities = set()
    
    valid_lines = 0
    duplicate_lines = 0
    
    print(f"--- 开始处理文件: {input_path} ---")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 构造一个不可变的 tuple 作为去重签名
                # 顺序：Head, HeadType, Relation, Tail, TailType
                triple_signature = (
                    data.get('head'), 
                    data.get('type'), 
                    data.get('relation'), 
                    data.get('tail'), 
                    data.get('tail_type')
                )
                
                # 检查是否完整 (防止空数据)
                if not all(triple_signature):
                    continue

                # 去重逻辑
                if triple_signature in seen_triples:
                    duplicate_lines += 1
                    print(triple_signature)
                    continue
                
                # 如果是新数据
                seen_triples.add(triple_signature)
                valid_lines += 1
                
                # 1. 写入新文件 (保持 jsonl 格式)
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # 2. 统计关系
                relation_counter[data['relation']] += 1
                
                # 3. 统计实体 (将头实体和尾实体都加入集合)
                # 格式: (实体名称, 实体类型)
                unique_entities.add((data['head'], data['type']))
                unique_entities.add((data['tail'], data['tail_type']))

            except json.JSONDecodeError:
                print(f"警告: 无法解析的行 -> {line}")
                continue

    # --- 统计节点类型的分布 ---
    entity_type_counter = Counter()
    for _, entity_type in unique_entities:
        entity_type_counter[entity_type] += 1

    # --- 打印统计报告 ---
    print(f"\n✅ 处理完成！结果已保存至: {output_path}")
    print(f"原始行数 (估算): {valid_lines + duplicate_lines}")
    print(f"有效三元组数 (去重后): {valid_lines}")
    print(f"移除重复行数: {duplicate_lines}")
    
    print("\n" + "="*40)
    print("📊 知识图谱数据统计")
    print("="*40)
    
    print(f"\n1. 实体节点统计 (Total Unique Nodes: {len(unique_entities)})")
    print("-" * 30)
    # 按数量降序打印
    for etype, count in entity_type_counter.most_common():
        print(f"{etype:<20} : {count} 个")
        
    print(f"\n2. 关系边统计 (Total Edges: {sum(relation_counter.values())})")
    print("-" * 30)
    for rel, count in relation_counter.most_common():
        print(f"{rel:<20} : {count} 条")
        
    print("="*40)

# ================= 使用示例 =================
if __name__ == "__main__":
    # 假设你的输入文件叫 triples.jsonl
    input_file = "triples_gpt-4o.jsonl"
    output_file = "triples_deduplicated.jsonl"
    
    
            
    clean_and_analyze_triples(input_file, output_file)