# src/Work4/config.py

# --- 渲染系统参数 ---
WINDOW_RES = (800, 600)    # 窗口分辨率

# --- 场景参数 ---
CAMERA_POS = [0.0, 0.0, 5.0]  # 摄像机位置
LIGHT_POS = [2.0, 3.0, 4.0]    # 点光源位置
LIGHT_COLOR = [1.0, 1.0, 1.0]  # 光源颜色
BACKGROUND_COLOR = [0.1, 0.2, 0.3]  # 背景颜色

# --- 球体参数 ---
SPHERE_CENTER = [-1.2, -0.2, 0.0]
SPHERE_RADIUS = 1.2
SPHERE_COLOR = [0.8, 0.1, 0.1]

# --- 圆锥参数 ---
CONE_APEX = [1.2, 1.2, 0.0]
CONE_BASE_Y = -1.4
CONE_RADIUS = 1.2
CONE_COLOR = [0.6, 0.2, 0.8]

# --- Phong 参数 ---
KA_DEFAULT = 0.2  # 环境光系数
KD_DEFAULT = 0.7  # 漫反射系数
KS_DEFAULT = 0.5  # 镜面高光系数
SHININESS_DEFAULT = 32.0  # 高光指数