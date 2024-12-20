## Problem3 : 参数 alpha

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的可复现性
np.random.seed(42)

# 参数定义
v = 0.5  # X_t 的长期均值
alpha_values = [0.01, 0.1, 0.5]  # X_t 的回归速度变化范围
sigma = 0.2  # X_t 的波动性
theta = 0.1  # S_t 回归到 X_t 的速度
sigma_hat1 = 0.15  # S_t 的波动性，与 B 相关
sigma_hat2 = 0.1  # S_t 的波动性，与 W 相关
x0 = 0.5  # X_t 的初始值
s0 = 0.5  # S_t 的初始值

# 时间设置
T = 10
N = 5000
dt = T / N

# 布朗运动
B = np.random.normal(0, np.sqrt(dt), N)
W = np.random.normal(0, np.sqrt(dt), N)

# 绘制参数 alpha 的影响
plt.figure(figsize=(15, 5))

# 绘制 X 的轨道图
plt.subplot(1, 3, 1)
for alpha in alpha_values:
    X = np.zeros(N+1)
    S = np.zeros(N+1)
    X[0] = x0
    S[0] = s0
    for t in range(1, N+1):
        dB = B[t-1]
        dW = W[t-1]
        X[t] = X[t-1] + alpha * (v - X[t-1]) * dt + sigma * dB
        S[t] = S[t-1] + theta * (X[t-1] - S[t-1]) * dt + sigma_hat1 * dB + sigma_hat2 * dW
    plt.plot(X, label=f'X, alpha={alpha}')
plt.title('Paths of X')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()

# 绘制 S 的轨道图
plt.subplot(1, 3, 2)
for alpha in alpha_values:
    X = np.zeros(N+1)
    S = np.zeros(N+1)
    X[0] = x0
    S[0] = s0
    for t in range(1, N+1):
        dB = B[t-1]
        dW = W[t-1]
        X[t] = X[t-1] + alpha * (v - X[t-1]) * dt + sigma * dB
        S[t] = S[t-1] + theta * (X[t-1] - S[t-1]) * dt + sigma_hat1 * dB + sigma_hat2 * dW
    plt.plot(S, label=f'S, alpha={alpha}', linestyle='--')
plt.title('Paths of S')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()

# 计算并绘制二阶混合变差
plt.subplot(1, 3, 3)
for alpha in alpha_values:
    X = np.zeros(N+1)
    S = np.zeros(N+1)
    X[0] = x0
    S[0] = s0
    for t in range(1, N+1):
        dB = B[t-1]
        dW = W[t-1]
        X[t] = X[t-1] + alpha * (v - X[t-1]) * dt + sigma * dB
        S[t] = S[t-1] + theta * (X[t-1] - S[t-1]) * dt + sigma_hat1 * dB + sigma_hat2 * dW
    qc = np.cumsum((X[1:] - X[:-1]) * (S[1:] - S[:-1]))
    plt.plot(qc, label=f'Quadratic Covariation, alpha={alpha}')
plt.title('Quadratic Covariation')
plt.xlabel('Time step')
plt.ylabel('Quadratic Covariation')
plt.legend()

plt.tight_layout()
plt.show()
