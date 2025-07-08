import pandas as pd
import numpy as np

def merge_txt_files(file1_path, file2_path, expected_length=4):
    try:
        # 读取文件1
        with open(file1_path, 'r', encoding='utf-8') as file1:
            lines1 = file1.readlines()
        
        # 读取文件2
        with open(file2_path, 'r', encoding='utf-8') as file2:
            lines2 = file2.readlines()
        
        # 合并两个文件的内容
        merged_lines = []
        for line1, line2 in zip(lines1, lines2):
            if line2.strip():  # 如果文件2对应行不为空
                merged_lines.append(line2.strip().split(','))
            else:  # 否则使用文件1的内容
                merged_lines.append(line1.strip().split(','))
        
        # 如果文件1的行数大于文件2，将剩余的行添加到合并结果中
        if len(lines1) > len(lines2):
            merged_lines.extend(line.strip().split(',') for line in lines1[len(lines2):])
    
        for i in range(len(merged_lines)):
            if len(merged_lines[i]) != expected_length:
                merged_lines[i] = lines1[i].strip().split(',')
        
        # 将合并的结果转换为NumPy数组
        merged_array = np.array(merged_lines, dtype=np.float32)
        
        # 将NumPy数组转为Pandas DataFrame并读取为gt_merge
        gt_merge = pd.DataFrame(merged_array).values
        
        return gt_merge

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"file1_path: {file1_path}")
        print(f"file2_path: {file2_path}")
        raise  # 重新引发异常以便进一步处理

# 示例文件路径
# file1_path = 'file1.txt'
# file2_path = 'file2.txt'

# # 合并文件并获取gt_merge
# gt_merge = merge_txt_files(file1_path, file2_path)
# print(gt_merge)
