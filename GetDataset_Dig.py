import cv2
import mediapipe as mp
import pandas as pd
import os
from Data_Preprocess import extract_processed_features_from_landmarks  # 引入预处理模块

# 初始化 MediaPipe Hands 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 启动摄像头
cap = cv2.VideoCapture(1)
all_data = []

# 数字手势标签映射（按下数字键 0-9）
label_map = {
    ord('0'): '0',
    ord('1'): '1',
    ord('2'): '2',
    ord('3'): '3',
    ord('4'): '4',
    ord('5'): '5',
    ord('6'): '6',
    ord('7'): '7',
    ord('8'): '8',
    ord('9'): '9'
}

print("开始采集手势数字数据，按数字键 0-9 为手势打标签，按 q 退出并保存数据。")
# 开始循坏采集，直到按键盘"q"结束
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    key = cv2.waitKey(10)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if key in label_map:
                features = extract_processed_features_from_landmarks(hand_landmarks.landmark)
                features.append(label_map[key])
                all_data.append(features)
                print(f"已采集：数字 {label_map[key]}，总样本数：{len(all_data)}")

    if key == ord('q'):
        print("退出采集，准备保存数据...")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        # 自动构造列名
        norm_cols = [f'nx{i}' for i in range(21)] + [f'ny{i}' for i in range(21)]
        dist_cols = [f'dist{i}' for i in range(5)]
        final_cols = norm_cols + dist_cols + ['angle', 'label']

        df_new = pd.DataFrame(all_data, columns=final_cols)
        filename = "hand_digits_dataset-[0-5].csv"

        # 合并已有数据（如果存在）
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                df_old = pd.read_csv(filename)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f"{filename} 是空的，直接用新数据覆盖。")
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}，总样本数：{len(df_combined)}")
        break

    cv2.imshow("Collect Hand Digits", frame)
