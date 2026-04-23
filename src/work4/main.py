# src/Work4/main.py
import taichi as ti
import taichi.ui as tiui

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)

# 导入我们自己写的模块
from .config import WINDOW_RES, KA_DEFAULT, KD_DEFAULT, KS_DEFAULT, SHININESS_DEFAULT
from .physics import render, image


def run():
    print("正在编译 GPU 内核，请稍候...")

    # 创建窗口
    window = tiui.Window("Experiment 4: Phong Lighting", res=WINDOW_RES)
    canvas = window.get_canvas()

    # UI 参数
    ka = KA_DEFAULT
    kd = KD_DEFAULT
    ks = KS_DEFAULT
    shininess = SHININESS_DEFAULT

    print("编译完成！请在弹出的窗口中调节参数。")

    # 渲染主循环
    while window.running:
        # 渲染
        render(ka, kd, ks, shininess)

        # 显示图像
        canvas.set_image(image)

        # UI
        with window.GUI.sub_window("Phong Parameters", 0.05, 0.05, 0.3, 0.4) as w:
            ka = w.slider_float("Ka (Ambient)", ka, 0.0, 1.0)
            kd = w.slider_float("Kd (Diffuse)", kd, 0.0, 1.0)
            ks = w.slider_float("Ks (Specular)", ks, 0.0, 1.0)
            shininess = w.slider_float("Shininess", shininess, 1.0, 128.0)

        window.show()


if __name__ == "__main__":
    run()