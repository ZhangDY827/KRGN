import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt
# 假设有一个64x3x3的卷积核
conv_kernel = np.random.randn(64, 3, 3)

# 选择前64个通道
selected_channels = conv_kernel[:64]

# 初始化64x64矩阵
kl_matrix = np.zeros((64, 64))

# 计算每对通道之间的KL散度
for i in range(64):
    for j in range(i+1, 64):
        # 应用softmax到每个KxK内核通道
        softmax_i = softmax(selected_channels[i].flatten())
        #print(softmax_i)
        softmax_j = softmax(selected_channels[j].flatten())
        
        # 计算KL散度
        #kl_divergence = entropy(softmax_i, softmax_j)
        kl_divergence = np.sum(softmax_i * np.log(softmax_i / softmax_j))
        # 填充矩阵
        kl_matrix[i, j] = kl_divergence
        kl_matrix[j, i] = kl_divergence

print("KL散度矩阵：")
normalized_matrix = (kl_matrix - np.min(kl_matrix)) / (np.max(kl_matrix) - np.min(kl_matrix))
print(kl_matrix)
plt.imshow(normalized_matrix, cmap='YlGnBu')
# 调整横轴和纵轴的数值间距（刻度间隔）
plt.xticks(np.arange(0, 64, 4))  # 设置横轴刻度间隔为8
plt.yticks(np.arange(0, 64, 4))  # 设置纵轴刻度间隔为8
##显示颜色条
plt.colorbar()
plt.xlabel('yxa1')
plt.ylabel('yxa2')
plt.title('Covariance Matrix Heatmap')
plt.savefig('heatmap.png')