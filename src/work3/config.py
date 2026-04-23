# src/work3/config.py

# --- 贝塞尔曲线实验参数 ---
WINDOW_RES = (800, 800)    # 窗口分辨率
NUM_SEGMENTS = 1000         # 曲线采样数
MAX_CONTROL_POINTS = 100    # 最大控制点数量
CONTROL_POINT_COLOR = (1.0, 0.0, 0.0)  # 控制点颜色 (红色)
CURVE_COLOR = (0.0, 1.0, 0.0)      # 曲线颜色 (绿色)
CONTROL_POINT_RADIUS = 2 / WINDOW_RES[1]