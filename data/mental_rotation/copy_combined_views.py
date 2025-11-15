"""
将每个structure目录中的combined_all_views.png文件复制到指定目录
并使用structure-*的命名格式重命名
"""

import os
import shutil
import re
import glob

# 源目录和目标目录
source_base_dir = r"D:\Projects\P3_CLER_Framework\Data_generation\data\images\mental_rotation\combined_views"
target_dir = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\data\mental_rotation"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 获取所有structure-*目录
structure_dirs = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d)) and d.startswith("structure-")]

# 正则表达式用于从目录名中提取序号
pattern = re.compile(r"structure-(\d+)")

# 计数器
success_count = 0
error_count = 0

print(f"开始处理，共找到 {len(structure_dirs)} 个structure目录...")

# 遍历每个structure目录
for structure_dir in structure_dirs:
    # 提取序号
    match = pattern.match(structure_dir)
    if not match:
        print(f"警告: '{structure_dir}' 格式不符，跳过")
        continue
    
    structure_num = match.group(1)
    
    # 源文件路径
    source_file = os.path.join(source_base_dir, structure_dir, "combined_all_views.png")
    
    # 目标文件路径及新文件名
    target_file = os.path.join(target_dir, f"structure-{structure_num}_combined_all_views.png")
    
    try:
        # 复制文件
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            success_count += 1
            print(f"已复制: {source_file} -> {target_file}")
        else:
            print(f"错误: 文件不存在 {source_file}")
            error_count += 1
    except Exception as e:
        print(f"错误: 复制 {source_file} 时出错: {e}")
        error_count += 1

print(f"\n处理完成! 成功: {success_count}, 失败: {error_count}")
