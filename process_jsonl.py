import json
import os

# 输入和输出文件路径
input_file = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\gemini_2_5_traindata\data\rubiks_cube\descrip\gemini\merged_rubiks_cube_colors_fixed.jsonl"
output_file = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\gemini_2_5_traindata\data\rubiks_cube\descrip\gemini\merged_rubiks_cube_colors_cleaned.jsonl"

# 计数器
total_lines = 0
removed_lines = 0

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        total_lines += 1
        try:
            # 尝试解析JSON
            data = json.loads(line.strip())
            
            # 检查raw_output是否为空或包含错误信息
            if 'raw_output' in data:
                raw_output = data.get('raw_output', '')
                if raw_output == '' or '请求失败' in raw_output or '任务终止' in raw_output:
                    removed_lines += 1
                    print(f"删除行 {total_lines}: {raw_output[:50]}...")
                    continue
            
            # 如果通过了检查，写入输出文件
            outfile.write(line)
        except json.JSONDecodeError:
            print(f"行 {total_lines} 不是有效的JSON，跳过")
            removed_lines += 1
            continue

print(f"处理完成。总行数: {total_lines}, 删除行数: {removed_lines}, 保留行数: {total_lines - removed_lines}")
