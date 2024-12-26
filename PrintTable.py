import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

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

    

for j in range(1, m + 1):
    # Условия на левой границе
    alpha[1] = 1
    beta[1] = 0

    for i in range(1, n):                           # Прямой ход
        phi = 5 * np.sin(t[j]) + u[i, j - 1] 
        alpha[i + 1] = B / (C - A * alpha[i])
        beta[i + 1] = (phi + A * beta[i]) / (C - A * alpha[i])

    # Обновления значения на правой границе
    u[-1, j] = (beta[-1] + 2 * h) / (1 + 7 * h - alpha[-1])

    for i in range(n - 1, -1, -1):                  # Обратный ход
        u[i, j] = alpha[i + 1] * u[i + 1, j] + beta[i + 1]

    # Обновления значения на левой границе
    u[0, j] = u[1, j]


df_full = pd.DataFrame(u.T, index=[f"t={ti:.2f}" for ti in t], columns=[f"x={xi:.2f}" for xi in x])

print("\nПолная таблица значений температуры на всех временных слоях:")
print(df_full)

# for j in range(1, 3):
#     A_m = np.zeros((5, 5))
#     A_m[0][0] = 1
#     A_m[1][0] = -1
#     A_m[3][4] = - 4 / 11
#     A_m[4][4] = 1
#     for i in range(1, 4):
#         A_m[i][i] = -C
#     for i in range(2, 5):
#         A_m[i][i - 1] = B
#     for i in range(3):
#          A_m[i][i + 1] = A


#     B_m = np.zeros(5)
#     for i in range(1, 4):
#         B_m[i] = -5 * np.sin(t[j]) - float(u[i + 1, 0]) / float(tau)
#     B_m[4] = 2 / 11

#     linSolve = np.linalg.solve(A_m.T, B_m)
#     print(linSolve)
  
