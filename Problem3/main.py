## Problem3(1)(2)问，绘制轨道

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的可复现性
np.random.seed(42)

# 参数定义
alpha = 0.1
v = 0.5
sigma = 0.2
theta = 0.1
sigma_hat1 = 0.15
sigma_hat2 = 0.1
x0 = 0.5
s0 = 0.5

# 时间设置
T = 10  # 总时间
N = 5000  # 时间步数
dt = T / N  # 时间步长

# 重复模拟以生成多条轨道
num_paths = 5

# 定义模拟路径的函数
def simulate_paths(N, num_paths, x0, s0, alpha, v, sigma, theta, sigma_hat1, sigma_hat2, dt):
    X_paths = np.zeros((num_paths, N+1))
    S_paths = np.zeros((num_paths, N+1))
    for i in range(num_paths):
        X_paths[i, 0] = x0
        S_paths[i, 0] = s0
        for t in range(1, N+1):
            dB = np.random.normal(0, np.sqrt(dt))
            dW = np.random.normal(0, np.sqrt(dt))
            X_paths[i, t] = X_paths[i, t-1] + alpha * (v - X_paths[i, t-1]) * dt + sigma * dB
            S_paths[i, t] = S_paths[i, t-1] + theta * (X_paths[i, t-1] - S_paths[i, t-1]) * dt + sigma_hat1 * dB + sigma_hat2 * dW
    return X_paths, S_paths

# 模拟路径
X_paths, S_paths = simulate_paths(N, num_paths, x0, s0, alpha, v, sigma, theta, sigma_hat1, sigma_hat2, dt)

# 绘制X和S的轨道图
plt.figure(figsize=(14, 6))

# 绘制X的路径
plt.subplot(1, 2, 1)
for i in range(num_paths):
    plt.plot(X_paths[i], label=f'X Path {i+1}')
plt.title('Paths of X')
plt.legend()

# 绘制S的路径
plt.subplot(1, 2, 2)
for i in range(num_paths):
    plt.plot(S_paths[i], label=f'S Path {i+1}')
plt.title('Paths of S')
plt.legend()

plt.tight_layout()
plt.show()

# 计算二阶混合变差
def quadratic_covariation(X, S):
    qc = np.cumsum((X[1:] - X[:-1]) * (S[1:] - S[:-1]))
    return qc

# 绘制每条轨道的二阶混合变差
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    qc = quadratic_covariation(X_paths[i], S_paths[i])
    plt.plot(qc, label=f'Quadratic Covariation Path {i+1}')
plt.title('Quadratic Covariation of X and S for Multiple Paths')
plt.legend()
plt.show()
