import taichi as ti
import taichi.ui as tiui

ti.init(arch=ti.gpu)

from .config import LEARNING_RATE_DEFAULT, TARGET_LIGHT_POS, WINDOW_RES
from .physics import (
    adam_update,
    compose_display,
    current_image,
    display_image,
    initialize_optimizer,
    light_pos,
    loss,
    render_current_and_loss,
    render_current_only,
    render_target,
    reset_loss,
)


def run():
    initialize_optimizer()
    render_target()

    window = tiui.Window("Experiment 6: Differentiable Rendering", res=WINDOW_RES)
    canvas = window.get_canvas()

    learning_rate = LEARNING_RATE_DEFAULT
    paused = False
    step = 0

    while window.running:
        for event in window.get_events(tiui.PRESS):
            if event.key == "p":
                paused = not paused
            elif event.key == "r":
                initialize_optimizer()
                step = 0

        if not paused:
            step += 1
            reset_loss()
            with ti.ad.Tape(loss):
                render_current_and_loss()
            adam_update(step, learning_rate)
        else:
            render_current_only()

        compose_display()
        canvas.set_image(display_image)

        lp = light_pos.to_numpy()
        with window.GUI.sub_window("Differentiable Renderer", 0.02, 0.02, 0.36, 0.32) as panel:
            learning_rate = panel.slider_float("Learning Rate", learning_rate, 0.001, 0.08)
            panel.text(f"Step: {step}")
            panel.text(f"Loss: {float(loss[None]):.6f}")
            panel.text(f"Light: ({lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f})")
            panel.text(f"Target: {TARGET_LIGHT_POS}")
            panel.text("Left: target, Right: optimized")
            panel.text("Press P: pause, R: reset")

        window.show()


if __name__ == "__main__":
    run()
