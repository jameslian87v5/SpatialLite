"""
修改JSONL文件中的question_type字段，将其从"descrip"改为"executive"
"""
import json
import os
from pathlib import Path

def update_jsonl_file(file_path):
    """
    读取JSONL文件，修改每个JSON对象中的question_type字段，然后保存回文件
    
    Args:
        file_path: JSONL文件路径
    """
    # 确保文件存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return False
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修改后的内容
    updated_lines = []
    modified_count = 0
    
    # 处理每一行
    for i, line in enumerate(lines):
        try:
            # 解析JSON
            data = json.loads(line.strip())
            
            # 检查并修改question_type字段
            if 'question_type' in data and data['question_type'] == 'descrip':
                data['question_type'] = 'executive'
                modified_count += 1
            
            # 将修改后的数据添加到结果中
            updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError:
            print(f"警告: 第 {i+1} 行不是有效的JSON，将保持原样")
            updated_lines.append(line)
    
    # 创建备份文件
    backup_path = file_path + '.bak'
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"已创建备份文件: {backup_path}")
    except Exception as e:
        print(f"创建备份文件时出错: {str(e)}")
        return False
    
    # 写入修改后的内容
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        print(f"成功修改了 {modified_count} 个对象的question_type字段")
        return True
    except Exception as e:
        print(f"写入文件时出错: {str(e)}")
        return False

def main():
    # 文件路径
    file_path = r"D:\Projects\P3_CLER_Framework\Data_generation\evaluation\outputs_o4\data\wood_slide\executive\o1\wood_slide_samples_20250323_192700_samples_20250505_090058.jsonl"
    
    print(f"开始处理文件: {file_path}")
    result = update_jsonl_file(file_path)
    
    if result:
        print("处理完成!")
    else:
        print("处理失败!")

if __name__ == "__main__":
    main()
