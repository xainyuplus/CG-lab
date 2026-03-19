import taichi as ti
import numpy as np
from . import config
from . import physics


# ---------------------------
# 主函数
# ---------------------------
def main():
    gui = ti.GUI("MVP Cube", (config.width, config.height))

    angle = 0.0

    while gui.running:
        gui.clear(0x0)

        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in ['a', ti.GUI.LEFT]:
                angle += config.step   # 向左
            elif e.key in ['d', ti.GUI.RIGHT]:
                angle -= config.step   # 向右

        # MVP 矩阵计算
        model = physics.get_model_matrix(angle)
        view = physics.get_view_matrix(config.eye_pos)
        projection = physics.get_projection_matrix(
            config.fov, 
            config.aspect_ratio, 
            config.z_near, 
            config.z_far
        )

        mvp = projection @ view @ model

        # 坐标变换
        pts = physics.transform(config.vertices, mvp)

        # 绘制立方体的所有边，使用蓝色
        for edge in config.edges:
            v1_idx, v2_idx = edge
            gui.line(pts[v1_idx], pts[v2_idx], radius=2, color=0x0000FF)

        gui.show()

if __name__ == "__main__":
    main()