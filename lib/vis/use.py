import visdom
import torch
from visdom_cus import VisHeatmap

# 初始化 visdom 实例
vis = visdom.Visdom(server='http://localhost', port=8097)

# 创建 VisHeatmap 实例
heatmap = VisHeatmap(vis, show_data=True, title="My Heatmap")

# 创建一个随机数据
data = torch.rand(10, 10)  # 创建一个10x10的随机矩阵

# 绘制热力图
heatmap.update(data, caption="Example Heatmap")

print("热力图已发送到 Visdom 界面，请在浏览器中查看！")
