import taichi as ti
import taichi.ui as tiui

ti.init(arch=ti.gpu)

from .config import CAMERA_LOOKAT, CAMERA_POS, LIGHT_POS, SUBSTEPS, WINDOW_RES
from .physics import initialize, positions, spring_indices, step


METHOD_NAMES = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]


def run():
    initialize()

    window = tiui.Window("Experiment 7: Mass-Spring Cloth", res=WINDOW_RES)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = tiui.Camera()
    camera.position(*CAMERA_POS)
    camera.lookat(*CAMERA_LOOKAT)

    method = 1
    paused = False

    while window.running:
        for event in window.get_events(tiui.PRESS):
            if event.key == "1":
                method = 0
                initialize()
            elif event.key == "2":
                method = 1
                initialize()
            elif event.key == "3":
                method = 2
                initialize()
            elif event.key == "p":
                paused = not paused
            elif event.key == "r":
                initialize()

        if not paused:
            for _ in range(SUBSTEPS):
                step(method)

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=tiui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=LIGHT_POS, color=(1.0, 1.0, 1.0))
        scene.particles(positions, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(positions, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        canvas.scene(scene)

        with window.GUI.sub_window("Control Panel", 0.02, 0.02, 0.38, 0.36) as panel:
            panel.text("Integration Method:")
            if panel.button(("[*] " if method == 0 else "[ ] ") + "Explicit Euler"):
                method = 0
                initialize()
            if panel.button(("[*] " if method == 1 else "[ ] ") + "Semi-Implicit Euler"):
                method = 1
                initialize()
            if panel.button(("[*] " if method == 2 else "[ ] ") + "Implicit Euler"):
                method = 2
                initialize()

            if panel.button("Resume Simulation" if paused else "Pause Simulation"):
                paused = not paused
            if panel.button("Reset Cloth"):
                initialize()

            panel.text("Keys: 1/2/3 switch, P pause, R reset")
            panel.text("Drag RMB to rotate camera")

        window.show()


if __name__ == "__main__":
    run()
