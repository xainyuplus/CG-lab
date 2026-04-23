# src/work3/main.py
import taichi as ti
import numpy as np

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)

# 导入我们自己写的模块
from .config import WINDOW_RES, NUM_SEGMENTS, MAX_CONTROL_POINTS, CONTROL_POINT_COLOR, CURVE_COLOR, CONTROL_POINT_RADIUS
from .physics import pixels, curve_points_field, gui_points, clear_pixels, draw_curve_kernel


def de_casteljau(points, t):
    """De Casteljau算法计算贝塞尔曲线上的点"""
    p = points[:]
    n = len(p)
    for r in range(1, n):
        for i in range(n - r):
            p[i] = (1 - t) * np.array(p[i]) + t * np.array(p[i + 1])
    return p[0]


def run():
    window = ti.ui.Window("Bézier Curve", WINDOW_RES)
    canvas = window.get_canvas()

    control_points = []

    while window.running:
        # 清空pixels
        clear_pixels()

        # 处理事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"添加控制点 {len(control_points)}: ({pos[0]:.3f}, {pos[1]:.3f})")
            elif e.key == 'c':
                control_points = []
                print("清空控制点")

        # 计算并绘制曲线
        if len(control_points) >= 2:
            curve_points_list = []
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                point = de_casteljau(control_points, t)
                curve_points_list.append(point)
            curve_points_field.from_numpy(np.array(curve_points_list))
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 绘制控制点（对象池技巧）
        gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0)
        for i, p in enumerate(control_points):
            gui_points_np[i] = p
        gui_points.from_numpy(gui_points_np)

        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=CONTROL_POINT_RADIUS, color=CONTROL_POINT_COLOR)
        
        

        window.show()


if __name__ == "__main__":
    run()