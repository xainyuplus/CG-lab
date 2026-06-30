import taichi as ti

from .config import *


image = ti.Vector.field(3, dtype=ti.f32, shape=WINDOW_RES)

CAMERA_POS_VEC = ti.Vector(CAMERA_POS)
BACKGROUND_COLOR_VEC = ti.Vector(BACKGROUND_COLOR)
LEFT_SPHERE_CENTER_VEC = ti.Vector(LEFT_SPHERE_CENTER)
RIGHT_SPHERE_CENTER_VEC = ti.Vector(RIGHT_SPHERE_CENTER)
RED_COLOR_VEC = ti.Vector(RED_COLOR)
MIRROR_TINT_VEC = ti.Vector(MIRROR_TINT)
LIGHT_COLOR_VEC = ti.Vector(LIGHT_COLOR)

GROUND_ID = 0
DIFFUSE_SPHERE_ID = 1
MIRROR_SPHERE_ID = 2


@ti.func
def intersect_sphere(ray_origin, ray_dir, center):
    oc = ray_origin - center
    half_b = oc.dot(ray_dir)
    c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS
    discriminant = half_b * half_b - c
    t = -1.0
    if discriminant >= 0.0:
        root = ti.sqrt(discriminant)
        near_t = -half_b - root
        far_t = -half_b + root
        if near_t > RAY_EPSILON:
            t = near_t
        elif far_t > RAY_EPSILON:
            t = far_t
    return t


@ti.func
def intersect_ground(ray_origin, ray_dir):
    t = -1.0
    if ti.abs(ray_dir.y) > 1e-6:
        candidate = (GROUND_Y - ray_origin.y) / ray_dir.y
        if candidate > RAY_EPSILON:
            t = candidate
    return t


@ti.func
def intersect_scene(ray_origin, ray_dir):
    closest_t = 1e10
    object_id = -1

    ground_t = intersect_ground(ray_origin, ray_dir)
    if ground_t > 0.0 and ground_t < closest_t:
        closest_t = ground_t
        object_id = GROUND_ID

    left_t = intersect_sphere(ray_origin, ray_dir, LEFT_SPHERE_CENTER_VEC)
    if left_t > 0.0 and left_t < closest_t:
        closest_t = left_t
        object_id = DIFFUSE_SPHERE_ID

    right_t = intersect_sphere(ray_origin, ray_dir, RIGHT_SPHERE_CENTER_VEC)
    if right_t > 0.0 and right_t < closest_t:
        closest_t = right_t
        object_id = MIRROR_SPHERE_ID

    if object_id < 0:
        closest_t = -1.0
    return closest_t, object_id


@ti.func
def surface_normal(point, object_id):
    normal = ti.Vector([0.0, 1.0, 0.0])
    if object_id == DIFFUSE_SPHERE_ID:
        normal = (point - LEFT_SPHERE_CENTER_VEC).normalized()
    elif object_id == MIRROR_SPHERE_ID:
        normal = (point - RIGHT_SPHERE_CENTER_VEC).normalized()
    return normal


@ti.func
def surface_color(point, object_id):
    color = RED_COLOR_VEC
    if object_id == GROUND_ID:
        checker = (ti.cast(ti.floor(point.x), ti.i32) + ti.cast(ti.floor(point.z), ti.i32)) & 1
        if checker == 0:
            color = ti.Vector([0.88, 0.88, 0.88])
        else:
            color = ti.Vector([0.08, 0.08, 0.08])
    return color


@ti.func
def is_in_shadow(point, normal, light_pos):
    to_light = light_pos - point
    light_distance = to_light.norm()
    shadow_dir = to_light / light_distance
    shadow_origin = point + normal * RAY_EPSILON
    blocker_t, _ = intersect_scene(shadow_origin, shadow_dir)
    return blocker_t > 0.0 and blocker_t < light_distance - RAY_EPSILON


@ti.func
def shade_diffuse(point, normal, view_dir, object_id, light_pos):
    base_color = surface_color(point, object_id)
    color = AMBIENT_STRENGTH * base_color

    if not is_in_shadow(point, normal, light_pos):
        light_dir = (light_pos - point).normalized()
        diffuse = DIFFUSE_STRENGTH * ti.max(normal.dot(light_dir), 0.0)
        reflected_light = (2.0 * normal.dot(light_dir) * normal - light_dir).normalized()
        specular = SPECULAR_STRENGTH * ti.pow(
            ti.max(reflected_light.dot(view_dir), 0.0), SHININESS
        )
        color += diffuse * base_color * LIGHT_COLOR_VEC
        color += specular * LIGHT_COLOR_VEC
    return color


@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32, max_bounces: ti.i32):
    light_pos = ti.Vector([light_x, light_y, light_z])
    aspect = WINDOW_RES[0] / WINDOW_RES[1]

    for i, j in image:
        screen_x = (2.0 * (ti.cast(i, ti.f32) + 0.5) / WINDOW_RES[0] - 1.0) * aspect
        screen_y = 2.0 * (ti.cast(j, ti.f32) + 0.5) / WINDOW_RES[1] - 1.0
        ray_origin = CAMERA_POS_VEC
        ray_dir = ti.Vector([screen_x, screen_y - 0.12, -1.8]).normalized()

        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])
        active = 1

        for bounce in ti.static(range(MAX_BOUNCES_LIMIT)):
            if active == 1 and bounce < max_bounces:
                hit_t, object_id = intersect_scene(ray_origin, ray_dir)
                if hit_t < 0.0:
                    final_color += throughput * BACKGROUND_COLOR_VEC
                    active = 0
                else:
                    point = ray_origin + hit_t * ray_dir
                    normal = surface_normal(point, object_id)
                    if object_id == MIRROR_SPHERE_ID:
                        ray_dir = (ray_dir - 2.0 * ray_dir.dot(normal) * normal).normalized()
                        ray_origin = point + normal * RAY_EPSILON
                        throughput *= MIRROR_REFLECTIVITY * MIRROR_TINT_VEC
                        if bounce + 1 >= max_bounces:
                            final_color += throughput * BACKGROUND_COLOR_VEC
                            active = 0
                    else:
                        view_dir = -ray_dir
                        final_color += throughput * shade_diffuse(
                            point, normal, view_dir, object_id, light_pos
                        )
                        active = 0

        image[i, j] = ti.min(ti.max(final_color, 0.0), 1.0)
