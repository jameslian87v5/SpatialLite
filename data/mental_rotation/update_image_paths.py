import json
import re
import os

def update_image_paths(json_file_path):
    """
    更新JSON文件中的图片路径，将格式从
    data\\images\\mental_rotation\\combined_views\\structure-1\\combined_all_views.png
    更改为
    structure_1_combined_all_views.png
    
    参数:
    json_file_path: JSON文件的路径
    """
    # 读取原始JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建临时文件用于写入修改后的内容
    temp_file_path = json_file_path + '.temp'
    
    print(f"开始处理文件: {json_file_path}")
    print(f"共有 {len(lines)} 行数据")
    
    modified_count = 0
    
    # 用于提取structure编号的正则表达式
    structure_pattern = re.compile(r'structure-(\d+)')
    
    with open(temp_file_path, 'w', encoding='utf-8') as f_out:
        for line in lines:
            # 将行解析为JSON
            try:
                sample = json.loads(line)
                
                # 获取原始图片路径
                original_image_path = sample.get('image', '')
                
                # 检查是否需要修改
                if 'combined_views\\structure-' in original_image_path and 'combined_all_views.png' in original_image_path:
                    # 提取structure编号
                    match = structure_pattern.search(original_image_path)
                    if match:
                        structure_num = match.group(1)
                        
                        # 生成新的图片路径
                        new_image_path = f'structure_{structure_num}_combined_all_views.png'
                        
                        # 更新样本中的图片路径
                        sample['image'] = new_image_path
                        
                        modified_count += 1
                        if modified_count <= 5:  # 只打印前5个修改示例
                            print(f"修改: {original_image_path} -> {new_image_path}")
                
                # 将修改后的样本写回文件
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"警告: 无法解析JSON行: {e}")
                # 保留原始行
                f_out.write(line)
    
    # 替换原始文件
    os.replace(temp_file_path, json_file_path)
    
    print(f"修改完成! 共修改了 {modified_count} 行数据")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # JSON文件完整路径
    json_file_path = os.path.join(script_dir, 'mental_rotation_samples.json')
    
    # 确保文件存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在: {json_file_path}")
    else:
        update_image_paths(json_file_path)
