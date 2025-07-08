import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save_individual_heatmaps(heatmaps_tensor, save_dir, type):
    """
    保存热图tensor为单独的图像文件并标注每个值的大小

    参数:
    heatmaps_tensor (torch.Tensor): 形状为 (N, 1, H, W) 的热图 tensor
    save_dir (str): 保存图像的目录
    base_filename (str): 基础文件名，文件名模式为 {base_filename}_{index}.png
    """
    # 将张量从 GPU 移动到 CPU（如果需要），并转换为 NumPy 数组
    heatmaps = heatmaps_tensor.cpu().detach().squeeze(1).numpy()
    
    # 获取热图数量
    num_heatmaps = heatmaps.shape[0]
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_heatmaps):
        heatmap = heatmaps[i]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap, cmap='viridis')
        
        # 标注每个值
        for (j, k), val in np.ndenumerate(heatmap):
            if type == 'sigmoid':
                ax.text(k, j, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)
            else:
                ax.text(k, j, f'{val:.0f}', ha='center', va='center', color='white', fontsize=8)
        
        ax.set_title(f'Heatmap {i + 1}')
        ax.axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=ax, shrink=0.5)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'heatmap_{i + 1}.png')
        plt.savefig(save_path)
        plt.close()

# 示例数据
# heatmaps_tensor = torch.rand(8, 1, 16, 16).cuda()  # 生成一个形状为 (8, 1, 16, 16) 的随机张量，并将其移动到 GPU 上

# 指定保存路径并调用函数
# save_path = 'path/to/save/heatmaps.png'
# save_heatmaps(heatmaps_tensor, save_path)
