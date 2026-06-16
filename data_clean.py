import re
import os

def clean_and_deduplicate_file(input_path, output_path):
    """
    读取一个txt文件，进行清理和去重，然后保存到新文件。
    清理规则：
    1. 移除所有完全空白的行。
    2. 将跨行的数字列表（如 `\n1)`）合并为一行（` 1)`）。
    3. 移除重复的行，只保留第一次出现的行。

    :param input_path: 输入文件的路径。
    :param output_path: 输出文件的路径。
    """
    print(f"--- 开始处理文件: {input_path} ---")

    # --- 步骤 1: 读取文件并处理格式 ---
    try:
        # 一次性读入整个文件内容，方便进行跨行替换
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print("UTF-8 解码失败，尝试使用 GBK 编码...")
        with open(input_path, 'r', encoding='gbk') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_path}'")
        return

    # 使用正则表达式替换 `\n` 后跟 `数字)` 的模式
    # 将其替换为一个空格后跟 `数字)`，避免文字直接连在一起
    # 例如：
    #   "检查门机系统"
    #   "1) 门刀"
    # 变为: "检查门机系统 1) 门刀"
    cleaned_content = re.sub(r'\n(\d+）)', r' \1', content)
    
    # 将处理后的文本按行分割成列表
    lines = cleaned_content.splitlines()
    print(f"原始文件（格式替换后）共有 {len(lines)} 行。")

    # --- 步骤 2: 去重并移除空行 ---
    unique_lines = []
    seen_lines = set()  # 使用集合来高效地检查重复项

    for line in lines:
        # 去除行首尾的空白字符（包括空格、制表符、换行符）
        stripped_line = line.strip()
        
        # 条件1: 必须是非空行
        # 条件2: 必须是第一次出现的行
        if stripped_line and stripped_line not in seen_lines:
            unique_lines.append(stripped_line)
            seen_lines.add(stripped_line)
            
    print(f"清理和去重后，剩余 {len(unique_lines)} 行。")

    # --- 步骤 3: 写入新文件 ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 每行后都加上换行符
            f.write('\n'.join(unique_lines))
        print(f"--- 处理完成，结果已保存至: {output_path} ---")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

# ================= 使用示例 =================

if __name__ == "__main__":
    # 定义输入和输出文件名
    input_file = "raw_data.txt"
    output_file = "cleaned_data.txt"


    clean_and_deduplicate_file(input_file, output_file)

  