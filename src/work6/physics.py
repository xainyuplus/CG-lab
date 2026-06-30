import taichi as ti

from .config import *


target_image = ti.Vector.field(3, dtype=ti.f32, shape=IMAGE_RES)
current_image = ti.Vector.field(3, dtype=ti.f32, shape=IMAGE_RES)
display_image = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
adam_m = ti.Vector.field(3, dtype=ti.f32, shape=())
adam_v = ti.Vector.field(3, dtype=ti.f32, shape=())
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

CAMERA_POS_VEC = ti.Vector(CAMERA_POS)
SPHERE_CENTER_VEC = ti.Vector(SPHERE_CENTER)
TARGET_LIGHT_POS_VEC = ti.Vector(TARGET_LIGHT_POS)
INITIAL_LIGHT_POS_VEC = ti.Vector(INITIAL_LIGHT_POS)
SPHERE_COLOR_VEC = ti.Vector(SPHERE_COLOR)
BACKGROUND_VEC = ti.Vector([BACKGROUND_INTENSITY, BACKGROUND_INTENSITY, BACKGROUND_INTENSITY])


@ti.func
def primary_ray(i, j):
    u = (ti.cast(i, ti.f32) + 0.5) / IMAGE_RES[0]
    v = (ti.cast(j, ti.f32) + 0.5) / IMAGE_RES[1]
    aspect = IMAGE_RES[0] / IMAGE_RES[1]
    x = (u - 0.5) * 1.15 * aspect
    y = (v - 0.5) * 1.15
    return ti.Vector([x, y, 1.0]).normalized()


@ti.func
def intersect_sphere(ray_origin, ray_dir):
    oc = ray_origin - SPHERE_CENTER_VEC
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS
    disc = b * b - 4.0 * c
    t = -1.0
    if disc >= 0.0:
        root = ti.sqrt(disc)
        t0 = (-b - root) * 0.5
        t1 = (-b + root) * 0.5
        if t0 > 0.0:
            t = t0
        elif t1 > 0.0:
            t = t1
    return t


@ti.func
def leaky_lambert(normal, light_dir):
    ndotl = normal.dot(light_dir)
    return ti.max(ndotl, LEAKY_ALPHA * ndotl)


@ti.func
def shade_raw(ray_dir, light):
    color = BACKGROUND_VEC
    t = intersect_sphere(CAMERA_POS_VEC, ray_dir)
    if t > 0.0:
        point = CAMERA_POS_VEC + t * ray_dir
        normal = (point - SPHERE_CENTER_VEC).normalized()
        light_dir = (light - point).normalized()
        intensity = AMBIENT + leaky_lambert(normal, light_dir)
        color = SPHERE_COLOR_VEC * intensity
    return color


@ti.func
def clamp_display(color):
    return ti.Vector([
        ti.min(1.0, ti.max(0.0, color.x)),
        ti.min(1.0, ti.max(0.0, color.y)),
        ti.min(1.0, ti.max(0.0, color.z)),
    ])


@ti.kernel
def initialize_optimizer():
    light_pos[None] = INITIAL_LIGHT_POS_VEC
    light_pos.grad[None] = ti.Vector([0.0, 0.0, 0.0])
    adam_m[None] = ti.Vector([0.0, 0.0, 0.0])
    adam_v[None] = ti.Vector([0.0, 0.0, 0.0])
    loss[None] = 0.0


@ti.kernel
def render_target():
    for i, j in target_image:
        ray_dir = primary_ray(i, j)
        target_image[i, j] = shade_raw(ray_dir, TARGET_LIGHT_POS_VEC)


@ti.kernel
def reset_loss():
    loss[None] = 0.0


@ti.kernel
def render_current_and_loss():
    for i, j in current_image:
        ray_dir = primary_ray(i, j)
        color = shade_raw(ray_dir, light_pos[None])
        current_image[i, j] = color
        diff = color - target_image[i, j]
        loss[None] += diff.dot(diff) / ti.cast(IMAGE_RES[0] * IMAGE_RES[1] * 3, ti.f32)


@ti.kernel
def render_current_only():
    for i, j in current_image:
        current_image[i, j] = shade_raw(primary_ray(i, j), light_pos[None])


@ti.kernel
def adam_update(step: ti.i32, learning_rate: ti.f32):
    grad = light_pos.grad[None]
    adam_m[None] = BETA1 * adam_m[None] + (1.0 - BETA1) * grad
    adam_v[None] = BETA2 * adam_v[None] + (1.0 - BETA2) * grad * grad

    bias_m = 1.0 - ti.pow(BETA1, ti.cast(step, ti.f32))
    bias_v = 1.0 - ti.pow(BETA2, ti.cast(step, ti.f32))
    m_hat = adam_m[None] / bias_m
    v_hat = adam_v[None] / bias_v

    light_pos[None] -= learning_rate * m_hat / (ti.sqrt(v_hat) + ADAM_EPS)


@ti.kernel
def compose_display():
    half_w = IMAGE_RES[0]
    for i, j in display_image:
        if i < half_w:
            display_image[i, j] = clamp_display(target_image[i, j])
        else:
            display_image[i, j] = clamp_display(current_image[i - half_w, j])
