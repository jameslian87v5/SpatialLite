import json
import re
import os

def process_cube_rolling_data(input_file, output_file):
    print(f"开始处理文件: {input_file}")
    
    # 直接作为JSONL格式读取（每行一个JSON对象）
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"无法解析行: {line[:50]}... 错误: {e}")
                    continue
    
    print(f"读取了 {len(data)} 条记录")
    
    # 处理每个记录
    processed_data = []
    for item in data:
        if "question" in item:
            # 正则表达式匹配完整的问题格式，包括前缀、路径和后缀
            # 这个正则表达式应该能够匹配到完整的路径部分，并将其完全替换
            # 格式: If the cube rolls according to the path shown "path1", "path2", ... "pathN", what color
            pattern = r'(If the cube rolls according to the path shown )"[^"]*"(?:,\s*"[^"]*")*,\s*(what color)'            
            
            # 进行替换，直接连接前缀和后缀，完全删除路径部分
            new_question = re.sub(pattern, r'\1\2', item["question"])
            
            # 如果替换成功（有变化）
            if new_question != item["question"]:
                item["question"] = new_question
            else:
                # 如果正则表达式没有匹配成功，尝试更通用的搜索方法
                start_phrase = "If the cube rolls according to the path shown \""
                if start_phrase in item["question"]:
                    start_idx = item["question"].find(start_phrase)
                    
                    if start_idx != -1:
                        # 找到起始位置
                        start_pos = start_idx + len("If the cube rolls according to the path shown ")
                        
                        # 寻找"what color"的位置，它通常在路径描述之后
                        end_marker = "what color"
                        end_pos = item["question"].find(end_marker, start_pos)
                        
                        if end_pos != -1:
                            # 重建问题文本，完全跳过路径部分
                            new_question = item["question"][:start_pos] + item["question"][end_pos:]
                            item["question"] = new_question
        
        processed_data.append(item)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理并保存了 {len(processed_data)} 条记录")
    print(f"处理完成，已保存到: {output_file}")
    return len(processed_data)

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\data\cube_rolling\cube_rolling_samples.json"
    output_file = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\data\cube_rolling\cube_rolling_samples_modified.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        exit(1)
    
    # 处理数据
    count = process_cube_rolling_data(input_file, output_file)
    print(f"成功处理并保存了 {count} 条记录")
