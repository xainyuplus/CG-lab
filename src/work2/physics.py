import numpy as np


def get_model_matrix(angle):
    """获取模型变换矩阵（旋转）"""
    rad = np.deg2rad(angle)

    model = np.array([
        [np.cos(rad), -np.sin(rad), 0, 0],
        [np.sin(rad),  np.cos(rad), 0, 0],
        [0,            0,           1, 0],
        [0,            0,           0, 1],
    ])
    
    return model


def get_view_matrix(eye_pos):
    """获取视图变换矩阵"""
    view = np.identity(4)
    view[0, 3] = -eye_pos[0]
    view[1, 3] = -eye_pos[1]
    view[2, 3] = -eye_pos[2]
    
    return view


def get_projection_matrix(eye_fov, aspect_ratio, z_near, z_far):
    """获取投影变换矩阵"""
    fov_rad = np.deg2rad(eye_fov)
    top = z_near * np.tan(fov_rad / 2)
    bottom = -top
    right = top * aspect_ratio
    left = -right
    
    persp_to_ortho = np.array([
        [z_near, 0,       0,               0],
        [0,      z_near,  0,               0],
        [0,      0,       z_near + z_far,  -z_near * z_far],
        [0,      0,       -1,              0]
    ])
        
    ortho = np.array([
        [2/(right - left), 0, 0, -(right + left)/(right - left)],
        [0, 2/(top - bottom), 0, -(top + bottom)/(top - bottom)],
        [0, 0, 2/(z_near - z_far), -(z_near + z_far)/(z_near - z_far)],
        [0, 0, 0, 1]
    ])
    
    projection = ortho @ persp_to_ortho
        
    return projection


def transform(vertices, mvp):
    """坐标变换：将顶点坐标从世界空间变换到屏幕空间"""
    transformed = []

    for v in vertices:
        v = mvp @ v

        # 齐次除法
        v = v / v[3]

        x = 0.5 * (v[0] + 1.0)
        y = 0.5 * (v[1] + 1.0)

        transformed.append([x, y])

    return np.array(transformed)
