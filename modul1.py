import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Размер сетки
n = 512
# Размер ширины дороги
width = 30

n += width * 2

# Генерация стартовых точек
p_start_up = (0, random.randint(width, n - width - 1))
p_start_l = (random.randint(width, (n - width - 1) // 2), 0)
p_start_r = (random.randint(width, (n - width - 1) // 2), n - 1)

p_start = random.choice([p_start_up, p_start_l, p_start_r])
p_end = (n - 1, random.randint(0, n - 1))


# Создание карты
map = np.zeros((n, n))


def get_biases(width, k):
    dx = (k / np.sqrt(1 + k**2)) * width
    dy = -(1 / np.sqrt(1 + k**2)) * width
    return width + dy - k * (width + dx), width - dy - k * (width - dx)

def get_square(width, k):
    eps_min = 1e-2
    eps_max = 10
    if np.abs(k) <= eps_min or np.abs(k) >= eps_max:
        print(k)
        return np.ones((2 * width, 2 * width))
    
    mask = np.zeros((2 * width, 2 * width))
    b = get_biases(width, k)
    # print(b)
    
    for x in range(2 * width):
        mask[x, :] = ((k * x + b[0] < np.arange(2 * width)) & (k * x + b[1] > np.arange(2 * width))).astype("int") 

    mask = mask.T
    # print(mask)
    # plt.imshow(mask, cmap='gray')
    # plt.title("Маска")
    # plt.show()
    return mask


def get_circle_mask(width):
    mask = np.zeros((2 * width, 2 * width))
    for x in range(2 * width):
        mask[x, :] = (((x - width) ** 2 + (np.arange(2 * width) - width) ** 2) <= width ** 2).astype("uint")
    # plt.imshow(mask, cmap='gray')
    # plt.title("Маска")
    # plt.show()
    return mask



# Функция для построения кривой Безье 3-го порядка
def bezier_curve(p0, p1, p2, p3, num_points=2*n):
    """
    Возвращает точки кривой Безье 3-го порядка для заданных контрольных точек.
    """
    t = np.linspace(0, 1, num_points)
    
    x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    dx = -3 * (1 - t)**2 * p0[0] - 6 * (1 - t) * t * p1[0] + 3 * (1 - t)**2 * p1[0] - 3 * t ** 2 * p2[0] + 6 * (1 - t) * t * p2[0] + 3 * t**2 * p3[0]
    y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    dy = -3 * (1 - t)**2 * p0[1] - 6 * (1 - t) * t * p1[1] + 3 * (1 - t)**2 * p1[1] - 3 * t ** 2 * p2[1] + 6 * (1 - t) * t * p2[1] + 3 * t**2 * p3[1]
    dy_dx = [dy[i]/dx_i if dx_i != 0 else 100 for i, dx_i in enumerate(dx)]
    return np.array([x, y, dy_dx])

# Генерация случайных контрольных точек
control_point_1 = (random.randint(width, n // 2), random.randint(width, n - width - 1))
control_point_2 = (random.randint(n // 2, n - width - 1), random.randint(width, n - width - 1))

# Построение кривой Безье
bezier_path = bezier_curve(p_start, control_point_1, control_point_2, p_end)

# Нанесение дороги на карту
def draw_road(map, path, width=5):
    """Рисует дорогу на карте, заполняя точки в заданной ширине."""
    for x, y, _ in zip(*path):
        x, y = int(round(x)), int(round(y))
        mask = get_circle_mask(width)
        map[max(0,x-width):min(map.shape[0],x+width), max(0,y-width): min(map.shape[1], y+width)] = \
            (map[max(0,x-width):min(map.shape[0],x+width), max(0,y-width): min(map.shape[1], y+width)] + mask[max(0, width - x):min(mask.shape[0], map.shape[0] - x + width), max(0,width-y): min(mask.shape[1], map.shape[1] - y + width)] > 0).astype("int")
    map *= 255
    map[path[0].astype('int'), path[1].astype('int')] = 128
    # plt.imshow(map, cmap='gray')
    # plt.title("Дорога с кривой Безье")
    # plt.show()
    # Обрезка краёв
    return map[width : n - width, width : n - width]
    

# Рисуем дорогу
res = draw_road(map, bezier_path, width)


# перепутали оси когда заполняли map
condition = np.argmin(np.abs(bezier_path[0] - n + 1 + width))
x_end = np.round(bezier_path[1][condition] - width)
y_end = np.round(bezier_path[0][condition] - width)

index = np.argmin((np.min(np.abs(bezier_path[0] - width)),
                  np.min(np.abs(bezier_path[1] - width)),
                  np.min(np.abs(bezier_path[1] - n + 1 + width)))
                 )

match index:
    case 0: # "up"
        condition = np.argmin(np.abs(bezier_path[0] - width))
    case 1: # "left"
        condition = np.argmin(np.abs(bezier_path[1] - width))
    case 2: # "right"
        condition = np.argmin(np.abs(bezier_path[1] - n + 1 + width))

x_start = np.round(bezier_path[1][condition] - width)
y_start = np.round(bezier_path[0][condition] - width)

print("start")
print(x_start, y_start)
print("end")
print(x_end, y_end)

# Визуализация
plt.imshow(res, cmap='gray')
plt.title("Дорога с кривой Безье")
plt.show()
