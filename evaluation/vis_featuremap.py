import matplotlib.pyplot as plt
import torch

def vis_rgb(img, title="rgb"):
    """
    img: (3,H,W) tensor
    """
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def vis_feat(feat, title="feat"):
    """
    feat: (C,H,W)
    用channel mean可视化
    """
    feat = feat.detach().cpu().mean(0)  # H,W
    plt.imshow(feat, cmap='viridis')
    # plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()


def vis_feat_enhanced(feat, title="feat", gamma=0.5):
    """
    gamma < 1 会增强亮区域（热点更明显）
    """
    feat = feat.detach().cpu().mean(0)  # H,W

    # min-max normalize
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)

    # gamma增强（关键）
    feat = feat ** gamma

    plt.imshow(feat, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()