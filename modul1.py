import torch
import random
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def get_biases(width, k):
    dx = (k / np.sqrt(1 + k**2)) * width
    dy = -(1 / np.sqrt(1 + k**2)) * width
    return width + dy - k * (width + dx), width - dy - k * (width - dx)


def get_square(width, k):
    eps_min = 1e-2
    eps_max = 10
    if np.abs(k) <= eps_min or np.abs(k) >= eps_max:
        # print(k)
        return np.ones((2 * width, 2 * width))
    
    mask = np.zeros((2 * width, 2 * width))
    b = get_biases(width, k)
    
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


# Нанесение дороги на карту
def draw_road(map, path, width=5):
    """Рисует дорогу на карте, заполняя точки в заданной ширине."""
    for x, y in zip(*path):
        x, y = int(round(x)), int(round(y))
        if not (0 <= x <= map.shape[0]) or not (0 <= y <= map.shape[1]):
            continue
        mask = get_circle_mask(width)
        map[max(0,x-width):min(map.shape[0],x+width), max(0,y-width): min(map.shape[1], y+width)] = \
            (map[max(0,x-width):min(map.shape[0],x+width), max(0,y-width): min(map.shape[1], y+width)] + mask[max(0, width - x):min(mask.shape[0], map.shape[0] - x + width), max(0,width-y): min(mask.shape[1], map.shape[1] - y + width)] > 0).astype("int")
    map *= 255
    # map[path[0].astype('int'), path[1].astype('int')] = 128
    # plt.imshow(map, cmap='gray')
    # plt.title("Дорога с кривой Безье")
    # plt.show()
    # Обрезка краёв
    n = map.shape[0]
    return map[width : n - width, width : n - width]


def bezier_curve(p0, p1, p2, p3, num_points):
    """
    Генерирует кривую Безье и продлевает её до границ карты с учетом второй производной.

    Parameters:
        p0, p1, p2, p3: tuple
            Контрольные точки кривой Безье.
        num_points: int
            Количество точек для генерации основной кривой.
        map_size: tuple
            Размеры карты (высота, ширина).

    Returns:
        np.array: Координаты кривой.
    """
    t = np.linspace(-0.1, 1.1, 2*num_points)
    
    # Оригинальная кривая Безье
    x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    
    return np.array([x, y])

def generate_road_mask(n, width, points):

    if points is None:
        p_start_up = (0, random.randint(0, n - 1))
        p_start_l = (random.randint(0, (n - 1) // 2), 0)
        p_start_r = (random.randint(0, (n - 1) // 2), n - 1)
        p_start, pos = random.choice([(p_start_up, "up"), (p_start_l, "left"), (p_start_r, "right")])
        p_end = (n - 1, random.randint(0, n - 1))
        points = [p_start, p_end]

    points = np.array(points) + width
    p_start, p_end = points[0], points[1]

    n += width * 2
    # Создание карты
    map = np.zeros((n, n))


    # Генерация случайных контрольных точек
    # Нужно грамтоно задать контрольные точки чтобы избежать горизонтальной касательной и вертикальной касаетльной
    # Нужно учсеть расположение points.
    min_p = max(n // 3, width)
    max_p = min((2 * n) // 3, n - width - 1)

    control_point_1 = (random.randint(min_p, max_p), random.randint(min_p, max_p))
    control_point_2 = (random.randint(min_p, max_p), random.randint(min_p, max_p))
    while np.linalg.norm(np.array(control_point_1)-np.array(control_point_2)) <= 2*width:
        control_point_2 = (random.randint(min_p, max_p), random.randint(min_p, max_p))

    # print(control_point_1)
    # print(control_point_2)

    # Построение кривой Безье
    bezier_path = bezier_curve(p_start, control_point_1, control_point_2, p_end, num_points=2 * n)

    # Рисуем дорогу
    res = draw_road(map, bezier_path, width)

    # res[control_point_1[0] - 5:control_point_1[0] + 5, control_point_1[1] - 5:control_point_1[1] + 5] = get_circle_mask(width=5) * 128
    # res[control_point_2[0] - 5:control_point_2[0] + 5, control_point_2[1] - 5:control_point_2[1] + 5] = get_circle_mask(width=5) * 128

    plt.imshow(res, cmap='gray')
    points -= width
    f_name = f"masks/w_{width} down_{points[1][1]}_{points[1][0]} {pos}_{points[0][1]}_{points[0][0]}.png"
    
    # Визуализация
    # plt.show()
    
    # Cохранение
    plt.axis('off')
    plt.imsave(f_name, res, cmap='gray')



def generate_roads(n, width, count, points=None):
    for w in width:
        for _ in range(count):
            generate_road_mask(n, w, points)


if __name__ == '__main__':
    n = 512
    width = [20, 30, 40, 50]
    
    # Генерация стартовых точек
    start_point = [0, random.randint(0, n - 1)]
    end_point = [n - 1, random.randint(0, n - 1)]
    point = [start_point, end_point]
    generate_roads(n, width, 10)
    
