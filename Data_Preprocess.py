import numpy as np
import math


def extract_processed_features_from_landmarks(landmarks):
    """
    输入：MediaPipe 的 21 个 landmark（含 x/y 坐标）
    输出：增强型特征（48维：归一化相对坐标 + 指尖距离 + 食指角度）
    """
    # 相对位置特征化
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    wrist = coords[0]
    rel_coords = coords - wrist

    # 手部标准缩放化
    xs, ys = coords[:, 0], coords[:, 1]
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1e-6
    norm_coords = rel_coords / max_range

    # 相对角度特征化
    dx, dy = coords[8][0] - wrist[0], coords[8][1] - wrist[1]
    angle = math.atan2(dy, dx)

    # 手指张开程度特征化
    tip_indices = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(coords[i] - wrist) for i in tip_indices]
    flat_norm = norm_coords.flatten().tolist()

    return flat_norm + distances + [angle]