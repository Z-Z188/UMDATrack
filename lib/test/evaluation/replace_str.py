import re

# 文件路径
file_path = '/nvme0n1/whj_file/models/Light-UAV-Track/lib/test/evaluation/uav123_hazedataset.py'

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 定位并替换 _get_sequence_info_list 函数中的 UAV123
pattern = r'(def _get_sequence_info_list\(self\):.*?)(def |\Z)'
replacement = lambda match: re.sub(r'data_seq/UAV123', 'data_seq/uav123_haze', match.group(1), flags=re.DOTALL)
updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 将修改后的内容写回文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(updated_content)

print("替换完成")
