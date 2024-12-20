## Problem2(2)(3)问：探究参数对轨道本身和二阶变差的影响

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的可复现性
np.random.seed(42)

# 基础参数设置
T = 1.0      # 总时间
N = 5000     # 时间步数
dt = T / N   # 时间步长

# 定义函数来模拟Ornstein-Uhlenbeck过程
def simulate_ou(alpha, v, sigma, x0=0.0):
    B_t = np.random.normal(0, np.sqrt(dt), N)
    X_t = np.zeros(N)
    X_t[0] = x0
    for t in range(1, N):
        X_t[t] = X_t[t-1] + alpha * (v - X_t[t-1]) * dt + sigma * B_t[t]
    return X_t

# 定义颜色列表
colors = ['blue', 'green', 'red']

# 绘制不同alpha的轨道和二阶变差
alphas = [1, 50, 200]
fig, axs = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle('Effect of Different Alpha on Process and Quadratic Variation')

for i, alpha in enumerate(alphas):
    X_t = simulate_ou(alpha, 1.0, 0.5)
    axs[0].plot(np.linspace(0, T, N), X_t, label=f'Alpha={alpha}', color=colors[i])
    axs[0].set_title('Ornstein-Uhlenbeck Process with Different Alphas')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X_t')
    axs[0].legend()

    quad_var = np.cumsum((X_t[1:] - X_t[:-1])**2)
    axs[1].plot(np.linspace(0, T, N-1), quad_var, label=f'Alpha={alpha}', color=colors[i])
    axs[1].set_title('Quadratic Variation with Different Alphas')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Quadratic Variation')
    axs[1].legend()

plt.show()

# 绘制不同v的轨道和二阶变差
vs = [0.5, 1.0, 1.5]
fig, axs = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle('Effect of Different V on Process and Quadratic Variation')

for i, v in enumerate(vs):
    X_t = simulate_ou(0.5, v, 0.5)
    axs[0].plot(np.linspace(0, T, N), X_t, label=f'V={v}', color=colors[i])
    axs[0].set_title('Ornstein-Uhlenbeck Process with Different Vs')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X_t')
    axs[0].legend()

    quad_var = np.cumsum((X_t[1:] - X_t[:-1])**2)
    axs[1].plot(np.linspace(0, T, N-1), quad_var, label=f'V={v}', color=colors[i])
    axs[1].set_title('Quadratic Variation with Different Vs')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Quadratic Variation')
    axs[1].legend()

plt.show()

# 绘制不同sigma的轨道和二阶变差
sigmas = [0.2, 0.5, 1.0]
fig, axs = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle('Effect of Different Sigma on Process and Quadratic Variation')

for i, sigma in enumerate(sigmas):
    X_t = simulate_ou(0.5, 1.0, sigma)
    axs[0].plot(np.linspace(0, T, N), X_t, label=f'Sigma={sigma}', color=colors[i])
    axs[0].set_title('Ornstein-Uhlenbeck Process with Different Sigmas')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X_t')
    axs[0].legend()

    quad_var = np.cumsum((X_t[1:] - X_t[:-1])**2)
    axs[1].plot(np.linspace(0, T, N-1), quad_var, label=f'Sigma={sigma}', color=colors[i])
    axs[1].set_title('Quadratic Variation with Different Sigmas')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Quadratic Variation')
    axs[1].legend()

plt.show()

# 绘制不同x0的轨道和二阶变差
x0s = [0.0, 0.5, 1.0]
fig, axs = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle('Effect of Different Initial Value X0 on Process and Quadratic Variation')

for i, x0 in enumerate(x0s):
    X_t = simulate_ou(0.5, 1.0, 0.5, x0)
    axs[0].plot(np.linspace(0, T, N), X_t, label=f'X0={x0}', color=colors[i])
    axs[0].set_title('Ornstein-Uhlenbeck Process with Different Initial Values')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X_t')
    axs[0].legend()

    quad_var = np.cumsum((X_t[1:] - X_t[:-1])**2)
    axs[1].plot(np.linspace(0, T, N-1), quad_var, label=f'X0={x0}', color=colors[i])
    axs[1].set_title('Quadratic Variation with Different Initial Values')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Quadratic Variation')
    axs[1].legend()

plt.show()
