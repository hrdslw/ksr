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

    # Прямой ход
    for i in range(1, n):                           
        phi = 5 * np.sin(t[j]) + u[i, j - 1] / float(tau)
        alpha[i + 1] = B / (C - A * alpha[i])
        beta[i + 1] = (phi + A * beta[i]) / (C - A * alpha[i])

    # Обновления значения на правой границе
    u[-1, j] = (beta[-1] + 2 * h) / (1 + 7 * h - alpha[-1])

    # Обратный ход
    for i in range(n - 1, -1, -1):                  
        u[i, j] = alpha[i + 1] * u[i + 1, j] + beta[i + 1]

    # Обновления значения на левой границе
    u[0, j] = u[1, j]


# Таблица
df_full = pd.DataFrame(u.T, index=[f"t[{i}]={ti:.2f}" for i, ti in enumerate(t)], columns=[f"x[{i}]={xi:.2f}" for i, xi in enumerate(x)])

print("\nТаблица значений температуры на всех временных слоях:")
print(df_full)
# Сохранение в файл
df_full.to_csv('table.csv', encoding='utf-8-sig', float_format='%.7f')

# Построение графика на j-ом временном слое
while True:
    print("Рисовать график на j-ом временном слое? (y/n) = (1/0)")
    if int(input()) == 1:
        while True:
            try:
                selected_j = int(input(f"Введите номер временного слоя для построения графика (0 - начальный, {m} - конечный): "))
                if 0 <= selected_j <= m:
                    break
                else:
                    print(f"Пожалуйста, введите число от 0 до {m}.")
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите целое число.")

        plt.figure(figsize=(10, 6))
        plt.plot(x, u[:, selected_j], marker='o', linestyle='-', color='b')
        plt.xlabel('x')
        plt.ylabel('Температура u(x, t)')
        plt.title(f'Распределение температуры на временном слое j={selected_j}, t={t[selected_j]:.2f}')
        plt.grid(True)
        plt.show()
    else:
        print("Остановка вывода графиков")
        break

# Построение 2D тепловой карты
X, Time = np.meshgrid(x, t)
plt.figure(figsize=(12, 6))
cp = plt.contourf(X, Time, u.T, 50, cmap='viridis')
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Тепловая карта распределения температуры')
plt.show()



#Построение 3D-графика
print("Построить 3D-график? (y/n) = (1/0)")
if int(input()) == 1:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Time, u.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.title('Распределение температуры')
    plt.show()
else:
    print("Не нарисовали 3D график")
