import cv2
import mediapipe as mp
import time
import socket
import joblib
from collections import deque, Counter
from Data_Preprocess import extract_processed_features_from_landmarks as DPP

# ============================= 加载两个模型 ==============================
direction_model = joblib.load("hand_direction_model.pkl")
digit_model = joblib.load("hand_digits_model.pkl")
digit_scaler = joblib.load("svm_scaler.pkl")
# ========================== 创建 Socket 服务器 ============================
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 12345))
server.listen(1)
print("等待客户端连接...")
conn, addr = server.accept()
conn.setblocking(False)  # 设置为非阻塞模式
print(f"客户端已连接: {addr}")
# ============================= 初始化 =============================
# 界面边界设置
border_size = 80
border_color = (128, 128, 128)
# 投票缓冲 + 冷却设置
direction_buffer = deque(maxlen=3)
digit_buffer = deque(maxlen=5)
last_sent_time = time.time()
cooldown = 0.01  # 初始为方向识别模式，快速响应
# 当前识别模式
current_mode = "direction"  # 初始为方向识别模式
# ========================= 主方法：手势识别结合版 =========================
def Run_CombinedRecognition():
    global current_mode, last_sent_time, cooldown
    # 摄像头与手部模型初始化
    vd = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7,max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils
    handLmsStyle = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=2)
    handconStyle = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=7, circle_radius=2)
    pTime = 0
    while True:
        try:# 检查是否有模式切换指令
            mode_switch = conn.recv(1024).decode()
            if mode_switch == "switch_to_direction":
                current_mode = "direction"
                cooldown = 0.001  # 方向模式快速响应
                print("切换到方向识别模式")
            elif mode_switch == "switch_to_digit":
                current_mode = "digit"
                cooldown = 0.5  # 数字模式稍慢但更准确
                print("切换到数字识别模式")
        except BlockingIOError:
            pass  # 没有数据可读
        except Exception as e:
            print(f"接收模式切换指令错误: {e}")
        ret, img = vd.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=border_color)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_RGB)

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS, handLmsStyle, handconStyle)
                features = DPP(handLms.landmark)
                if current_mode == "direction":
                    # 方向识别模式
                    predicted = direction_model.predict([features])[0]
                    direction_buffer.append(predicted)
                    most_common, count = Counter(direction_buffer).most_common(1)[0]
                    current_time = time.time()
                    if count >= 2 and current_time - last_sent_time >= cooldown:
                        try:
                            conn.send(most_common.encode())
                            last_sent_time = current_time
                            print(f"已发送方向: {most_common}")
                        except:
                            break
                    cv2.putText(img, f"Direction: {most_common}", (border_size + 20, border_size + 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif current_mode == "digit":
                    # 数字识别模式
                    features = features[:-1]  # 移除角度特征
                    try:
                        features_scaled = digit_scaler.transform([features])
                        predicted_digit = digit_model.predict(features_scaled)[0]
                        digit_buffer.append(predicted_digit)
                        most_common_digit, count = Counter(digit_buffer).most_common(1)[0]

                        current_time = time.time()
                        if count >= 3 and current_time - last_sent_time >= cooldown:
                            try:
                                conn.send(str(most_common_digit).encode())
                                last_sent_time = current_time
                                print(f"已发送数字: {most_common_digit}")
                            except:
                                break

                        cv2.putText(img, f"Digit: {most_common_digit}", (border_size + 20, border_size + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    except Exception as e:
                        print(f"数字预测时出错: {str(e)}")
                        continue

        # 显示FPS和当前模式
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (border_size + 10, border_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f"Mode: {current_mode}", (WIDTH - border_size - 200, border_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow('Hand Recognition (Combined)', img)

        if cv2.waitKey(1) == ord('q'):
            break

    vd.release()
    cv2.destroyAllWindows()
    server.close()


if __name__ == '__main__':
    WIDTH = 640 + border_size * 2  # 假设摄像头宽度为640
    Run_CombinedRecognition()
