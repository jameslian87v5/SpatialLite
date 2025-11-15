#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重命名脚本：去掉当前目录下文件名中的"(1)"部分，支持将PNG图像重命名为JPG，并删除文件名中的空格
"""

import os
import re

def rename_files_remove_duplicate_marker():
    """扫描当前目录，去掉文件名中的'(1)'标记"""
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"扫描目录: {current_dir}")
    
    # 统计计数
    total_files = 0
    renamed_files = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(current_dir):
        total_files += 1
        
        # 检查文件名是否包含"(1)"
        if "(1)" in filename:
            # 使用正则表达式去掉"(1)"部分
            new_name = re.sub(r'\(1\)', '', filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(os.path.join(current_dir, new_name)):
                print(f"警告: 无法重命名 '{filename}' -> '{new_name}' (目标文件已存在)")
                continue
            
            # 重命名文件
            old_path = os.path.join(current_dir, filename)
            new_path = os.path.join(current_dir, new_name)
            
            try:
                os.rename(old_path, new_path)
                renamed_files += 1
                print(f"重命名: '{filename}' -> '{new_name}'")
            except Exception as e:
                print(f"错误: 重命名 '{filename}' 失败: {str(e)}")
    
    # 打印统计结果
    print(f"\n处理完成!")
    print(f"扫描文件总数: {total_files}")
    print(f"重命名文件数: {renamed_files}")

def rename_png_to_jpg():
    """将PNG图像文件重命名为JPG格式"""
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"扫描目录: {current_dir}")
    
    # 统计计数
    total_files = 0
    renamed_files = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(current_dir):
        if filename.lower().endswith('.png'):
            total_files += 1
            
            # 创建新的文件名，将.png替换为.jpg
            new_name = filename[:-4] + '.jpg'
            
            # 检查新文件名是否已存在
            if os.path.exists(os.path.join(current_dir, new_name)):
                print(f"警告: 无法重命名 '{filename}' -> '{new_name}' (目标文件已存在)")
                continue
            
            # 重命名文件
            old_path = os.path.join(current_dir, filename)
            new_path = os.path.join(current_dir, new_name)
            
            try:
                os.rename(old_path, new_path)
                renamed_files += 1
                print(f"重命名: '{filename}' -> '{new_name}'")
            except Exception as e:
                print(f"错误: 重命名 '{filename}' 失败: {str(e)}")
    
    # 打印统计结果
    print(f"\n处理完成!")
    print(f"扫描PNG文件总数: {total_files}")
    print(f"重命名文件数: {renamed_files}")

def remove_spaces_from_filenames():
    """删除文件名中的空格"""
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"扫描目录: {current_dir}")
    
    # 统计计数
    total_files = 0
    renamed_files = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(current_dir):
        total_files += 1
        
        # 检查文件名是否包含空格
        if ' ' in filename:
            # 去掉文件名中的空格
            new_name = filename.replace(' ', '')
            
            # 检查新文件名是否已存在
            if os.path.exists(os.path.join(current_dir, new_name)):
                print(f"警告: 无法重命名 '{filename}' -> '{new_name}' (目标文件已存在)")
                continue
            
            # 重命名文件
            old_path = os.path.join(current_dir, filename)
            new_path = os.path.join(current_dir, new_name)
            
            try:
                os.rename(old_path, new_path)
                renamed_files += 1
                print(f"重命名: '{filename}' -> '{new_name}'")
            except Exception as e:
                print(f"错误: 重命名 '{filename}' 失败: {str(e)}")
    
    # 打印统计结果
    print(f"\n处理完成!")
    print(f"扫描文件总数: {total_files}")
    print(f"重命名文件数: {renamed_files}")

if __name__ == "__main__":
    # 执行重命名操作
    print("1. 去除文件名中的(1)标记")
    rename_files_remove_duplicate_marker()
    
    print("\n2. 将PNG图像重命名为JPG")
    rename_png_to_jpg()
    
    print("\n3. 删除文件名中的空格")
    remove_spaces_from_filenames()