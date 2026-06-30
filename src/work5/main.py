import taichi as ti
import taichi.ui as tiui

ti.init(arch=ti.gpu)

from .config import LIGHT_POS_DEFAULT, MAX_BOUNCES_DEFAULT, WINDOW_RES
from .physics import image, render


def run():
    window = tiui.Window("Experiment 5: Whitted Ray Tracing", res=WINDOW_RES)
    canvas = window.get_canvas()

    light_x, light_y, light_z = LIGHT_POS_DEFAULT
    max_bounces = float(MAX_BOUNCES_DEFAULT)

    while window.running:
        render(light_x, light_y, light_z, int(round(max_bounces)))
        canvas.set_image(image)

        with window.GUI.sub_window("Ray Tracing Controls", 0.02, 0.02, 0.32, 0.38) as panel:
            light_x = panel.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = panel.slider_float("Light Y", light_y, 0.2, 8.0)
            light_z = panel.slider_float("Light Z", light_z, -5.0, 6.0)
            max_bounces = panel.slider_float("Max Bounces", max_bounces, 1.0, 5.0)
            panel.text(f"Active bounces: {int(round(max_bounces))}")

        window.show()


if __name__ == "__main__":
    run()
