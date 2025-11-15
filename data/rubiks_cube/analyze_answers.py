import json
import os
from collections import Counter

# 文件路径
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubiks_cube_samples.json")

# 初始化计数器
color_counter = Counter()
number_counter = Counter()
total_samples = 0

# 颜色列表
colors = ["red", "green", "blue", "yellow", "orange", "grey"]

# 读取并解析JSON文件
with open(file_path, "r", encoding="utf-8") as f:
    # 文件包含多个JSON对象，每行一个
    lines = f.readlines()
    
    for line in lines:
        if line.strip():  # 跳过空行
            try:
                data = json.loads(line)
                answer = data.get("answer")
                total_samples += 1
                
                # 判断答案类型并计数
                if isinstance(answer, str) and answer.lower() in colors:
                    color_counter[answer.lower()] += 1
                elif isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
                    # 确保是整数
                    if isinstance(answer, str):
                        answer = int(answer)
                    number_counter[answer] += 1
                else:
                    print(f"未知答案类型: {answer}, 类型: {type(answer)}")
            
            except json.JSONDecodeError:
                print(f"无法解析JSON行: {line}")
            except Exception as e:
                print(f"处理数据时出错: {e}")

# 计算总数
total_colors = sum(color_counter.values())
total_numbers = sum(number_counter.values())

# 打印统计结果
print("\n===== 魔方答案分布统计 =====")
print(f"总样本数: {total_samples}")

print("\n颜色答案分布:")
print("  【相对于总样本】   【相对于颜色答案】")
for color in colors:
    count = color_counter[color]
    percentage_of_total = (count / total_samples) * 100 if total_samples > 0 else 0
    percentage_of_colors = (count / total_colors) * 100 if total_colors > 0 else 0
    print(f"  {color}: {count} ({percentage_of_total:.2f}%) | {percentage_of_colors:.2f}%")

print("\n数字答案分布:")
print("  【相对于总样本】   【相对于数字答案】")
for number in sorted(number_counter.keys()):
    count = number_counter[number]
    percentage_of_total = (count / total_samples) * 100 if total_samples > 0 else 0
    percentage_of_numbers = (count / total_numbers) * 100 if total_numbers > 0 else 0
    print(f"  {number}: {count} ({percentage_of_total:.2f}%) | {percentage_of_numbers:.2f}%")

# 汇总
print("\n分类汇总:")
print(f"颜色答案总数: {total_colors} ({(total_colors / total_samples) * 100:.2f}%)")
print(f"数字答案总数: {total_numbers} ({(total_numbers / total_samples) * 100:.2f}%)")

