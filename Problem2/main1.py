## Problem2(1)问 ：绘制多条轨道和二阶变差轨道

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的可复现性
np.random.seed(42)

# 参数设置
alpha = 0.5  # 回归速度
v = 1.0      # 长期均值
sigma = 0.5  # 波动性
x0 = 0.0     # 初始值
T = 1.0      # 总时间
N = 5000     # 时间步数
dt = T / N   # 时间步长
num_trajectories = 5  # 轨迹数量

# 生成布朗运动
B_t = np.random.normal(0, np.sqrt(dt), (N, num_trajectories))

# 模拟Ornstein-Uhlenbeck过程
X_t = np.zeros((N, num_trajectories))
X_t[0, :] = x0
for t in range(1, N):
    X_t[t, :] = X_t[t-1, :] + alpha * (v - X_t[t-1, :]) * dt + sigma * B_t[t, :]

# 绘制过程X的轨道
plt.figure(figsize=(10, 5))
for i in range(num_trajectories):
    plt.plot(np.linspace(0, T, N), X_t[:, i], label=f'Ornstein-Uhlenbeck Process {i+1}')
plt.title('Ornstein-Uhlenbeck Process')
plt.xlabel('Time')
plt.ylabel('X_t')
plt.legend()
plt.show()

# 计算二阶变差
quad_var = np.cumsum((X_t[1:, :] - X_t[:-1, :])**2, axis=0)

# 绘制二阶变差
plt.figure(figsize=(10, 5))
for i in range(num_trajectories):
    plt.plot(np.linspace(0, T, N-1), quad_var[:, i], label=f'Quadratic Variation {i+1}', color='red')
plt.title('Quadratic Variation of the Process')
plt.xlabel('Time')
plt.ylabel('Quadratic Variation')
plt.legend()
plt.show()
