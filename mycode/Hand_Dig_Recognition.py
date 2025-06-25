import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
from collections import deque, Counter
from Data_Preprocess import extract_processed_features_from_landmarks as DPP  # 引入预处理模块

# 加载训练好的SVM模型和标准化器
svm_model = joblib.load("hand_digits_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

# 界面边界设置
border_size = 80
border_color = (128, 128, 128)

# 投票缓冲
digit_buffer = deque(maxlen=10)  # 取最近10次预测结果做投票

# 每个数字对应一种颜色
digit_colors = {
    0: (0, 0, 255),     # 鲜艳纯红（BGR顺序，红色）
    1: (0, 255, 0),     # 鲜艳纯绿
    2: (255, 0, 0),     # 鲜艳纯蓝
    3: (255, 0, 255),   # 鲜艳纯紫（红蓝混合）
    4: (128, 0, 255),   # 鲜艳紫蓝
    5: (255, 0, 128),   # 鲜艳紫红
    6: (0, 255, 255),   # 鲜艳青色（蓝绿混合）
    7: (255, 128, 0),   # 鲜艳橙红（虽然偏橙，但不偏黄）
    8: (0, 128, 255),   # 鲜艳天蓝
    9: (255, 255, 255),  # 纯白色（9用白色，看得最清）
    'None': (64, 64, 64) # 深灰色
}

def Run_HandDigitRecognition():
    # 摄像头与手部模型初始化
    vd = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.9,  # 提高检测置信度
        min_tracking_confidence=0.8,   # 提高跟踪置信度
        max_num_hands=1                # 只检测一只手
    )
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

        current_digit = 'None'  # 默认是None

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS, handLmsStyle, handconStyle)

                features = DPP(handLms.landmark)[:-1]

                try:
                    features_scaled = scaler.transform([features])
                    predicted_digit = svm_model.predict(features_scaled)[0]
                    digit_buffer.append(predicted_digit)

                    most_common_digit, count = Counter(digit_buffer).most_common(1)[0]
                    current_digit = most_common_digit

                except Exception as e:
                    print(f"预测时出错: {str(e)}")
                    continue

        # 获取颜色
        color = digit_colors.get(current_digit, (255, 255, 255))

        # 在左上角绘制
        cv2.putText(img, f"Digit: {current_digit}", (border_size + 20, border_size + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 显示FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (border_size + 10, border_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Hand Digit Recognition (SVM)', img)

        if cv2.waitKey(1) == ord('q'):
            break


    vd.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Run_HandDigitRecognition()
