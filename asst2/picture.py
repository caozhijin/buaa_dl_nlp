from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 数据
T_values = [5, 10, 15, 20, 25]
K_values = [20, 100, 500, 1000, 3000]

# Table 1: Mode为char时的数据
accuracy_char = np.array([
    [0.116, 0.157, 0.181, 0.223, 0.174],
    [0.244, 0.388, 0.439, 0.477, 0.537],
    [0.375, 0.563, 0.693, 0.715, 0.828],
    [0.422, 0.628, 0.744, 0.839, 0.837],
    [0.55, 0.733, 0.822, 0.919, 0.878]
])

# Table 2: Mode为word时的数据
accuracy_word = np.array([
    [0.127, 0.167, 0.186, 0.221, 0.243],
    [0.300, 0.400, 0.549, 0.645, 0.703],
    [0.483, 0.661, 0.757, 0.759, 0.805],
    [0.531, 0.704, 0.729, 0.823, 0.898],
    [0.618, 0.763, 0.78, 0.831, 0.898]
])

# 绘图
plt.figure(figsize=(14, 8))

# 不同主题数量T对分类性能的影响
for i, K in enumerate(K_values):
    plt.plot(T_values, accuracy_char[i], marker='o', label=f'K={K} (char)')
    plt.plot(T_values, accuracy_word[i], marker='x', label=f'K={K} (word)')

plt.title('Impact of Topic Number T on Classification Performance')
plt.xlabel('Number of Topics (T)')
plt.ylabel('Average Classification Accuracy')
plt.grid(True)
plt.show()

# 以“词”和以“字”为token时分类结果的差异

# 创建图形和3D子图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 为“char”模式生成面
X, Y = np.meshgrid(K_values, T_values)
ax.plot_surface(X, Y, accuracy_char.T, color='blue', alpha=0.5, label='char')

# 为“word”模式生成面
ax.plot_surface(X, Y, accuracy_word.T, color='red', alpha=0.5, label='word')

ax.set_xlabel('Number of Tokens (K)')
ax.set_ylabel('Number of Topics (T)')
ax.set_zlabel('Average Classification Accuracy')
ax.legend()
plt.show()


# 不同文本长度K下主题模型性能的变化
plt.figure(figsize=(14, 8))
for i, T in enumerate(T_values):
    plt.plot(K_values, accuracy_char[i, :], marker='o', label=f'T={T} (char)')
    plt.plot(K_values, accuracy_word[i, :], marker='x', label=f'T={T} (word)')

plt.title('Performance Variation of Topic Model with Different Text Lengths K')
plt.xlabel('Number of Tokens (K)')
plt.ylabel('Average Classification Accuracy')
plt.legend(title='Number of Topics (T)')
plt.grid(True)
plt.show()