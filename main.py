import numpy as np
import matplotlib.pyplot as plt


# Ввод размеров сетки
print("Введите размеры сетки")
n = int(input("n: "))
m = int(input("m: "))

# Параметры задачи
L = 1  # Длина стержня
T = 1000  # Время
h = L / n
tau = T / m

# Сетки по x и t
x = np.linspace(0, L, n + 1)
t = np.linspace(0, T, m + 1)

# Начальное распределение температуры
u = np.zeros((n + 1, m + 1))
u[:, 0] = 1 - x**2

# Коэффициенты для метода прогонки
A = 9 / h**2    # 9 - коэффициент температуропроводимости
B = 9 / h**2
C = 1 / tau + 18 / h**2

# Инициализация массивов alpha и beta
alpha = np.zeros(n + 1)
beta = np.zeros(n + 1)


def straight_running():     # Прямой ход

    for i in range(1, n):
        phi = 5 * np.sin(t[j]) + u[i, j - 1]
        alpha[i + 1] = B / (C - A * alpha[i])
        beta[i + 1] = (phi + A * beta[i]) / (C - A * alpha[i])


def reverse_running():      # Обратный ход

    for i in range(n - 1, -1, -1):
        u[i, j] = alpha[i + 1] * u[i + 1, j] + beta[i + 1]


for j in range(1, m + 1):
    # Условия на левой границе
    alpha[1] = 1
    beta[1] = 0

    straight_running()
    # Обновления значения на правой границе
    u[-1, j] = (beta[-1] + 2 * h) / (1 + 7 * h - alpha[-1])

    reverse_running()
    # Обновления значения на левой границе
    u[0, j] = u[1, j]


# Построение 3D-графика
X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.title('Распределение температуры')
plt.show()