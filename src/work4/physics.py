# src/Work4/physics.py
import taichi as ti
from .config import *

# 数据结构：图像缓冲区
image = ti.Vector.field(3, dtype=float, shape=WINDOW_RES)

# 转换为向量
CAMERA_POS_VEC = ti.Vector(CAMERA_POS)
LIGHT_POS_VEC = ti.Vector(LIGHT_POS)
LIGHT_COLOR_VEC = ti.Vector(LIGHT_COLOR)
BACKGROUND_COLOR_VEC = ti.Vector(BACKGROUND_COLOR)
SPHERE_CENTER_VEC = ti.Vector(SPHERE_CENTER)
SPHERE_COLOR_VEC = ti.Vector(SPHERE_COLOR)
CONE_APEX_VEC = ti.Vector(CONE_APEX)
CONE_COLOR_VEC = ti.Vector(CONE_COLOR)

@ti.func
def intersect_sphere(ray_origin, ray_dir):
    """计算光线与球体的交点"""
    oc = ray_origin - SPHERE_CENTER_VEC
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS
    discriminant = b * b - 4 * a * c
    t = -1.0
    if discriminant >= 0:
        t1 = (-b - ti.sqrt(discriminant)) / (2 * a)
        t2 = (-b + ti.sqrt(discriminant)) / (2 * a)
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
    return t

@ti.func
def intersect_cone(ray_origin, ray_dir):
    """计算光线与圆锥的交点"""
    # 圆锥方程：从顶点向下，底面在y = CONE_BASE_Y
    apex = CONE_APEX_VEC
    height = apex.y - CONE_BASE_Y
    radius = CONE_RADIUS

    # 变换到圆锥坐标系
    ro = ray_origin - apex
    rd = ray_dir

    # 圆锥参数
    k = radius / height
    k = k * k

    a = rd.x * rd.x + rd.z * rd.z - k * rd.y * rd.y
    b = 2 * (ro.x * rd.x + ro.z * rd.z - k * ro.y * rd.y)
    c = ro.x * ro.x + ro.z * ro.z - k * ro.y * ro.y

    discriminant = b * b - 4 * a * c
    t = -1.0
    if discriminant >= 0:
        t1 = (-b - ti.sqrt(discriminant)) / (2 * a)
        t2 = (-b + ti.sqrt(discriminant)) / (2 * a)

        # 检查t1
        if t1 > 0:
            p = ro + t1 * rd
            if p.y <= 0 and p.y >= -height:
                t = t1

        # 检查t2，如果t还没有设置
        if t < 0 and t2 > 0:
            p = ro + t2 * rd
            if p.y <= 0 and p.y >= -height:
                t = t2
    return t

@ti.func
def get_sphere_normal(point):
    """获取球体表面法向量"""
    return (point - SPHERE_CENTER_VEC).normalized()

@ti.func
def get_cone_normal(point):
    """获取圆锥表面法向量"""
    apex = CONE_APEX_VEC
    height = apex.y - CONE_BASE_Y
    radius = CONE_RADIUS
    k = radius / height

    p = point - apex
    # 法向量计算：对于圆锥，法向量垂直于母线
    # 圆锥方程：x^2 + z^2 = k^2 * y^2
    # 梯度：(2x, -2k^2 y, 2z)
    grad = ti.Vector([2 * p.x, -2 * k * k * p.y, 2 * p.z])
    return grad.normalized()

@ti.func
def phong_shading(point, normal, view_dir, ka, kd, ks, shininess, obj_color):
    """Phong着色计算"""
    light_dir = (LIGHT_POS_VEC - point).normalized()
    reflect_dir = (2 * normal.dot(light_dir) * normal - light_dir).normalized()

    ambient = ka * LIGHT_COLOR_VEC * obj_color
    diffuse = kd * ti.max(0.0, normal.dot(light_dir)) * LIGHT_COLOR_VEC * obj_color
    specular = ks * ti.pow(ti.max(0.0, reflect_dir.dot(view_dir)), shininess) * LIGHT_COLOR_VEC

    return ambient + diffuse + specular

@ti.kernel
def render(ka: float, kd: float, ks: float, shininess: float):
    """渲染kernel"""
    for i, j in image:
        # 计算像素坐标
        x = (i - WINDOW_RES[0] / 2) / (WINDOW_RES[0] / 2)
        y = (j - WINDOW_RES[1] / 2) / (WINDOW_RES[1] / 2)
        z = -1.0  # 近平面

        # 光线方向
        ray_dir = ti.Vector([x, y, z]).normalized()
        ray_origin = CAMERA_POS_VEC

        # 求交
        t_sphere = intersect_sphere(ray_origin, ray_dir)
        t_cone = intersect_cone(ray_origin, ray_dir)

        color = BACKGROUND_COLOR_VEC

        if t_sphere > 0 or t_cone > 0:
            if t_sphere > 0 and (t_cone < 0 or t_sphere < t_cone):
                # 击中球体
                hit_point = ray_origin + t_sphere * ray_dir
                normal = get_sphere_normal(hit_point)
                view_dir = (CAMERA_POS_VEC - hit_point).normalized()
                color = phong_shading(hit_point, normal, view_dir, ka, kd, ks, shininess, SPHERE_COLOR_VEC)
            elif t_cone > 0:
                # 击中圆锥
                hit_point = ray_origin + t_cone * ray_dir
                normal = get_cone_normal(hit_point)
                view_dir = (CAMERA_POS_VEC - hit_point).normalized()
                color = phong_shading(hit_point, normal, view_dir, ka, kd, ks, shininess, CONE_COLOR_VEC)

        image[i, j] = color