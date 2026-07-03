import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_bool_tensor(tensor, save_path, title=None):
    """
    可视化一个 m x n 的 bool tensor 并保存为图片。
    
    参数:
        tensor (torch.Tensor or np.ndarray): 形状为 (m, n) 的布尔张量。
        save_path (str): 图片保存路径。
        title (str, optional): 图片标题。
    """
    # 转换为 numpy 数组并确保是布尔类型
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)
    
    if data.dtype != bool:
        data = data.astype(bool)
        
    if data.ndim != 2:
        raise ValueError(f"输入张量必须是 2D 的，但得到的是 {data.ndim}D")

    m, n = data.shape
    
    # 根据张量比例动态设置画布大小
    # 假设每个元素大约占一定的显示空间
    fig_width = max(8, n / 20)
    fig_height = max(8, m / 20)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 使用 'binary' colormap: True 为黑色 (0), False 为白色 (1)
    # 或者 'gray': True 为白色 (1), False 为黑色 (0)
    # 这里我们用 'binary'，True (1) 会显示为黑色，False (0) 为白色
    # 如果想反过来，可以使用 'binary_r'
    plt.imshow(data, cmap='binary', interpolation='nearest')
    
    if title:
        plt.title(title, fontsize=16)
    
    plt.axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"张量可视化已保存至: {save_path}")

if __name__ == "__main__":
    # 测试代码
    m, n = 50, 80
    test_tensor = torch.rand(m, n) > 0.5
    visualize_bool_tensor(test_tensor, "test_bool_tensor.png", title="Example Bool Tensor Visualization")
