import os
import json

class SmartChunker:
    def __init__(self, input_path, output_path, max_lines=50, max_chars=1000, overlap_lines=10):
        """
        :param input_path: 输入文件路径
        :param output_path: 输出文件路径
        :param max_lines: 每个块允许的最大行数 (软限制)
        :param max_chars: 每个块允许的最大字符数 (硬限制，模拟 max_token)
        :param overlap_lines: 块与块之间重叠的行数
        """
        self.input_path = input_path
        self.output_path = output_path
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.overlap_lines = overlap_lines

    def read_lines(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"找不到文件: {self.input_path}")
            
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                # 去除首尾空白，保留非空行
                return [line.rstrip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(self.input_path, 'r', encoding='gbk') as f:
                return [line.rstrip() for line in f.readlines() if line.strip()]

    def process_and_save(self):
        lines = self.read_lines()
        total_lines = len(lines)
        chunks = []
        
        start_index = 0
        
        print(f"--- 开始处理 ---")
        print(f"输入文件: {self.input_path}")
        print(f"限制条件: Max Lines={self.max_lines}, Max Chars={self.max_chars}, Overlap={self.overlap_lines}")
        
        while start_index < total_lines:
            current_chunk_lines = []
            current_char_count = 0
            end_index = start_index
            
            # --- 贪婪构建当前块 ---
            while end_index < total_lines:
                line = lines[end_index]
                line_len = len(line)
                
                # 检查条件 1: 是否超过最大行数 (如果是第一行则强制加入，防止死循环)
                if len(current_chunk_lines) >= self.max_lines and len(current_chunk_lines) > 0:
                    break
                
                # 检查条件 2: 是否超过最大字数 (如果是第一行则强制加入)
                if current_char_count + line_len > self.max_chars and len(current_chunk_lines) > 0:
                    break
                
                current_chunk_lines.append(line)
                current_char_count += line_len
                end_index += 1
            
            # 将当前块存入列表
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append({
                "id": len(chunks) + 1,
                "text": chunk_text,
                "lines": len(current_chunk_lines),
                "chars": current_char_count,
                "range": f"{start_index + 1} - {end_index}"
            })
            
            # --- 计算下一个块的起始位置 (处理重叠) ---
            # 正常情况下，下一个块从 (当前结束位置 - 重叠行数) 开始
            next_start = end_index - self.overlap_lines
            
            # 边界保护：
            # 1. 如果当前块太短（少于重叠行数），强制至少前进一步，防止死循环
            # 2. 如果已经到了文件末尾，跳出循环
            if next_start <= start_index:
                start_index += 1
            else:
                start_index = next_start
                
            if end_index >= total_lines:
                break

        # --- 保存到文件 ---
        self._save_to_file(chunks)
        print(f"--- 处理完成: 共生成 {len(chunks)} 个块，已保存至 {self.output_path} ---")
        return chunks


    def _save_to_file(self, chunks):
        """
        将分块结果写入 JSONL 格式 (每行一个合法的 JSON 对象)
        优势：自动处理所有特殊字符转义，Python 读取极其方便
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # 构造字典
                record = {
                    "id": chunk['id'],
                    "range": chunk['range'],
                    "char_count": chunk['chars'],
                    "content": chunk['text']  # 这里放入原始带换行的文本即可
                }
                
                # json.dumps 会自动将 content 里的换行符转义为 \n
                # ensure_ascii=False 保证中文正常显示，而不是 \uXXXX
                json_line = json.dumps(record, ensure_ascii=False)
                
                f.write(json_line + "\n")


if __name__ == "__main__":
    # 1. 定义输入输出路径
    input_file = "cleaned_data.txt"   # 你的源文件
    output_file = "elevator_chunks.jsonl" # 结果保存文件

  
    # 2. 初始化并运行
    # max_chars=1000: 大约对应 500-800 个 token，适合大多数 LLM
    # max_lines=50: 即使字数很少，到了50行也强制截断，保持语义紧凑
    chunker = SmartChunker(
        input_path=input_file, 
        output_path=output_file, 
        max_lines=50, 
        max_chars=500, 
        overlap_lines=3
    )
    
    chunker.process_and_save()