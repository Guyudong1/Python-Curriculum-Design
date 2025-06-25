import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
from collections import deque, Counter
from Data_Preprocess import extract_processed_features_from_landmarks as DPP  # 引入预处理模块

# 加载训练好的方向识别模型
clf = joblib.load("hand_direction_model.pkl")

# 界面边界设置
border_size = 80
border_color = (128, 128, 128)

# 投票缓冲
direction_buffer = deque(maxlen=3)

# 不同方向对应不同颜色
direction_colors = {
    "Up": (0, 0, 255),         # 红色
    "Down": (0, 255, 0),       # 绿色
    "Left": (255, 0, 0),       # 蓝色
    "Right": (128, 0, 128),    # 深紫色
    "None": (64, 64, 64)   # 灰色（备用）
}

def Run_HandRecognition():
    # 摄像头与手部模型初始化
    vd = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    handLmsStyle = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=2)
    handconStyle = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=7, circle_radius=2)
    pTime = 0

    while True:
        ret, img = vd.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)

        img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                 cv2.BORDER_CONSTANT, value=border_color)

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_RGB)

        most_common = "None"  # 默认无方向

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS, handLmsStyle, handconStyle)

                # 提取特征并预测方向
                delta = DPP(handLms.landmark)
                predicted = clf.predict([delta])[0]
                direction_buffer.append(predicted)

                most_common, _ = Counter(direction_buffer).most_common(1)[0]

        # 取颜色
        color = direction_colors.get(most_common, (255, 255, 255))  # 默认白色

        # 画 Direction
        cv2.putText(img, f"Direction: {most_common}", (border_size + 20, border_size + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 显示 FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (border_size + 10, border_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Hand Direction Recognition', img)

        if cv2.waitKey(1) == ord('q'):
            break

    vd.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Run_HandRecognition()
