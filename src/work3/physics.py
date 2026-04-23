# src/work3/physics.py
import taichi as ti
from .config import *

# 数据结构定义
pixels = ti.Vector.field(3, dtype=float, shape=WINDOW_RES)  # 最终画面
curve_points_field = ti.Vector.field(2, dtype=float, shape=NUM_SEGMENTS + 1)  # 曲线坐标
gui_points = ti.Vector.field(2, dtype=float, shape=MAX_CONTROL_POINTS)  # 控制点对象池


@ti.kernel
def clear_pixels():
    """清空pixels缓冲区"""
    for i, j in pixels:
        pixels[i, j] = [0.0, 0.0, 0.0]


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """GPU绘制曲线内核：对单色曲线做双线性插值写入"""
    W = WINDOW_RES[0]
    H = WINDOW_RES[1]
# 直接写入像素（不做抗锯齿）
#x, y = curve_points_field[i] 
#px = int(x * WINDOW_RES[0]) 
#py = int(y * WINDOW_RES[1]) 
#if 0 <= px < WINDOW_RES[0] and 0 <= py < WINDOW_RES[1]: pixels[px, py] = [0.0, 1.0, 0.0] # 绿色

    for i in range(n):
        x, y = curve_points_field[i]

        # 归一化坐标 -> 像素连续坐标
        fx = x * (W - 1)
        fy = y * (H - 1)

        x0 = int(fx)
        y0 = int(fy)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = fx - x0
        dy = fy - y0

        # 双线性权重
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        # 分配到四个邻近像素，只写绿色通道
        if 0 <= x0 < W and 0 <= y0 < H:
            pixels[x0, y0][1] = max(pixels[x0, y0][1], w00)

        if 0 <= x1 < W and 0 <= y0 < H:
            pixels[x1, y0][1] = max(pixels[x1, y0][1], w10)

        if 0 <= x0 < W and 0 <= y1 < H:
            pixels[x0, y1][1] = max(pixels[x0, y1][1], w01)

        if 0 <= x1 < W and 0 <= y1 < H:
            pixels[x1, y1][1] = max(pixels[x1, y1][1], w11)