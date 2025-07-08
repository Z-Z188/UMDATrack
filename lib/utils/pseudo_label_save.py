# import os

# def write_to_txt(file_path, line_number, tensor):
#     # 读取现有内容
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 如果文件行数小于指定行数，则添加空行
#     # while len(lines) < line_number:
#     #     lines.append("\n")
#     if len(lines) < line_number:
#         raise ValueError("the line is out of the txt! file_path:{}".format(file_path))
    
#     # 将tensor转换为所需的格式化字符串
#     tensor_str = ','.join(f'{value:.4f}' for value in tensor) + '\n'
    
#     # 在指定行写入内容
#     lines[line_number - 1] = tensor_str
    
#     # 写回文件
#     with open(file_path, 'w') as file:
#         file.writelines(lines)

# 示例调用
# file_path = '指定的文件路径/文件名.txt'
# line_number = 5  # 要写入的行号
# text = '这是写入的内容'

# write_to_txt(file_path, line_number, text)
# print(f'已在{file_path}文件的第{line_number}行写入内容: {text}')

# import threading

# # 创建全局锁对象
# file_lock = threading.Lock()

# def write_to_txt(file_path, line_number, tensor):
#     # 获取锁
#     with file_lock:
#         # 读取现有内容
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
        
#         # 如果文件行数小于指定行数，则添加空行
#         if len(lines) < line_number:
#             raise ValueError(f"指定行数 {line_number} 超出文件范围！文件路径: {file_path}")
        
#         # 将 tensor 转换为所需的格式化字符串
#         tensor_str = ','.join(f'{value:.4f}' for value in tensor) + '\n'
        
#         # 在指定行写入内容
#         lines[line_number - 1] = tensor_str
        
#         # 写回文件
#         with open(file_path, 'w') as file:
#             file.writelines(lines)

from filelock import FileLock, Timeout
import os

def write_to_txt(file_path, line_number, tensor, timeout=60):
    lock_path = file_path + '.lock'
    lock = FileLock(lock_path, timeout=timeout)
    
    try:
        # 获取锁
        with lock:
            # 读取现有内容
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件 {file_path} 不存在。")
            
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 如果文件行数小于指定行数，则抛出错误
            if len(lines) < line_number:
                raise ValueError(f"指定行数 {line_number} 超出文件范围！文件路径: {file_path}")
            
            # 将 tensor 转换为所需的格式化字符串
            tensor_str = ','.join(f'{value:.4f}' for value in tensor) + '\n'
            
            # 在指定行写入内容
            lines[line_number - 1] = tensor_str
            
            # 写回文件
            with open(file_path, 'w') as file:
                file.writelines(lines)
    except Timeout:
        print(f"在 {timeout} 秒内无法获取文件锁。请重试。")
