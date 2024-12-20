import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的可复现性
np.random.seed(42)

# 参数设置
T = 1.0          # 总时间
N = 2000         # 时间点数量
dt = T / N       # 时间步长
num_paths = 5    # 生成的轨道数量

# 生成布朗运动的轨道
BM_paths = np.zeros((N, num_paths))
for i in range(1, N):
    dB = np.random.normal(0, np.sqrt(dt), num_paths)
    BM_paths[i] = BM_paths[i-1] + dB

# 计算一阶变差
first_variation = np.abs(BM_paths[1:] - BM_paths[:-1])

# 计算二阶变差
second_variation = (BM_paths[1:] - BM_paths[:-1])**2

for i in range(2, N-1):
    first_variation[i] = first_variation[i] + first_variation[i-1]
    second_variation[i] = second_variation[i] + second_variation[i-1]


# 绘制布朗运动轨道
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(np.linspace(0, T, N), BM_paths[:, i], label=f'Path {i+1}')
plt.title('Brownian Motion Paths')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 绘制一阶变差
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(np.linspace(0, T, N-1), first_variation[:, i], label=f'Path {i+1}')
plt.title('First Variation of Brownian Motion')
plt.xlabel('Time')
plt.ylabel('First Variation')
plt.legend()
plt.show()

# 绘制二阶变差
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(np.linspace(0, T, N-1), second_variation[:, i], label=f'Path {i+1}')
plt.title('Second Variation of Brownian Motion')
plt.xlabel('Time')
plt.ylabel('Second Variation')
plt.legend()
plt.show()
