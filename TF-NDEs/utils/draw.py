import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap_seaborn(tensor, title="Heatmap", save_path="heatmap.png"):
    tensor = tensor.detach().cpu().numpy()
    df = pd.DataFrame(tensor)
    plt.figure(figsize=(10, 8))  # 可选：设置图像大小
    sns.heatmap(df, cmap="Greys")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()



def plot_signal_components(yh_tensor, yl_tensor, save_path):
    """可视化信号分量并保存图像"""
    
    
    # 转换为numpy数组
    yh_data = _to_numpy(yh_tensor)
    yl_data = _to_numpy(yl_tensor)
    
    plt.figure(figsize=(10, 5))
    plt.plot(yh_data, 'r-', linewidth=1.5, alpha=0.8, label='low Yh')
    plt.plot(yl_data, 'b--', linewidth=1.5, alpha=0.8, label='high Yl')

    # 图表美化设置
    plt.title('signal compare', fontsize=14, pad=20)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('value', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

def _to_numpy(tensor):
    """将PyTorch张量转换为numpy数组（处理批量数据中的第一个样本）"""
    if isinstance(tensor, (list, tuple)):  # 处理多返回值的特殊情况
        tensor = tensor[0]
    if tensor.dim() == 3:  # [B, T, N] 格式
        return tensor[2, 0, :].detach().cpu().numpy()
    elif tensor.dim() == 2:  # [T, N] 格式
        return tensor[:, 0].detach().cpu().numpy()
    else:
        return tensor[0].detach().cpu().numpy()  # 默认处理

