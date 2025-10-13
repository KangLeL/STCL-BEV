import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(map):
    """map: [1, H, W]"""
    heatmap = np.squeeze(map)

    plt.imshow(heatmap, cmap='hot')  # 'hot' 是常见的热图颜色映射，也可以选择 'viridis', 'plasma' 等
    plt.colorbar()  # 显示颜色条
    plt.axis('off')  # 可选：不显示坐标轴
    plt.show()