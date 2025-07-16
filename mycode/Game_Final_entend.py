import pygame
import random
import time
import socket
import threading
import sys
import os
from Human_Vs_AI_DIFScreen import main_game as human_vs_ai_game
from Human_VS_AI_SameScreen import main_game as human_vs_ai_same_screen

# ================================== 全局变量 ==================================
RECOGNITION_MODE = "direction"  # 初始模式
last_digit = None
socket_lock = threading.Lock()
hand_detected = False
DIGIT_HOLD_TIME = 3.0  # 需要保持3秒
current_digit = None
digit_start_time = 0
confirmed_digit = None
GESTURE_IMAGE = None  # 手势图示
GESTURE_IMAGE_RECT = None
TIMEOUT = 20  # 20秒超时
last_interaction_time = time.time()
game_paused = False
pause_start_time = 0
pause_duration = 0

# ============================== 初始化socket连接 ==============================
def init_connection():
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2.0)  # 连接超时2秒
        client.connect(('localhost', 12345))
        print("成功连接到手势识别服务")
        return client
    except Exception as e:
        print(f"连接失败: {e}")
        return None

client = None  # 初始化为None，在main中初始化

# ============================== 安全发送带重试 ==============================
def safe_send_with_retry(data, max_retries=3, retry_delay=0.5):
    global client
    for attempt in range(max_retries):
        if safe_send(data):
            return True
        print(f"发送失败，尝试重连 ({attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        client = init_connection()
    return False

# ============================== 加载手势图片 ==============================
def load_gesture_image():
    global GESTURE_IMAGE, GESTURE_IMAGE_RECT
    try:
        image_path = os.path.join("images", "gestures.png")
        GESTURE_IMAGE = pygame.image.load(image_path)
        GESTURE_IMAGE = pygame.transform.scale(GESTURE_IMAGE, (200, 200))
        GESTURE_IMAGE_RECT = GESTURE_IMAGE.get_rect()
        GESTURE_IMAGE_RECT.bottomright = (WIDTH - 20, HEIGHT - 20)
    except Exception as e:
        print(f"无法加载手势图片: {e}")
        GESTURE_IMAGE = None

# ================================== 游戏核心 ===================================
# 初始化pygame
pygame.init()
global replay_frames
replay_frames = []  # 存储回放帧数据
GRID_SIZE = 50
GRID_COUNT = 12
BORDER_WIDTH = 80
WIDTH, HEIGHT = GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2, GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2
# 在文件顶部添加窗口大小定义
WINDOW_SINGLE = (WIDTH, HEIGHT)
WINDOW_DUAL = (1200, 600)
# 颜色定义
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
DARK_GREEN = (0, 90, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BORDER_COLOR = (50, 50, 50)
# 方向定义
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = {"Up": UP, "Down": DOWN, "Left": LEFT, "Right": RIGHT}
DIGITS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
# 界面初始化
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("手势控制贪吃蛇")
# 字体初始化
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 36)
# 加载手势图片
load_gesture_image()

# =============================== 贪吃蛇主要类 ===============================
class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        base_x = BORDER_WIDTH
        base_y = HEIGHT - BORDER_WIDTH - GRID_SIZE

        self.body = [
            [base_x, base_y],
            [base_x - GRID_SIZE, base_y],
            [base_x - 2 * GRID_SIZE, base_y]
        ]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.grow = False
        self.speed = 4

    def move(self):
        self.direction = self.next_direction
        head = [
            self.body[0][0] + self.direction[0] * GRID_SIZE,
            self.body[0][1] + self.direction[1] * GRID_SIZE
        ]

        # 碰撞检测
        if (head in self.body[1:] or
                head[0] < BORDER_WIDTH or
                head[0] >= WIDTH - BORDER_WIDTH or
                head[1] < BORDER_WIDTH or
                head[1] >= HEIGHT - BORDER_WIDTH):
            return False

        self.body.insert(0, head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        return True

    def change_direction(self, new_dir):
        # 禁止180度转弯
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.next_direction = new_dir

    def eat_food(self, food_pos):
        if self.body[0] == food_pos:
            self.grow = True
            return True
        return False

# ============================= 安全socket通信 ==============================
def is_socket_alive(sock):
    try:
        if sock:
            sock.getpeername()
            return True
    except:
        pass
    return False

def safe_send(data):
    global client
    with socket_lock:
        if not is_socket_alive(client):
            client = init_connection()
            if not client:
                return False

        try:
            client.sendall(data.encode())
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            client = None
            return False

def safe_receive():
    global client
    with socket_lock:
        if not is_socket_alive(client):
            client = init_connection()
            if not client:
                return None

        try:
            client.settimeout(0.1)
            data = client.recv(1024).decode()
            return data if data else None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"接收失败: {e}")
            client = None
            return None

def close_socket():
    global client
    with socket_lock:
        if client:
            try:
                client.close()
            except:
                pass
            client = None

# ============================= 控制接收线程 =============================
def control_thread(snake):
    global RECOGNITION_MODE, hand_detected, last_digit, current_digit, digit_start_time, confirmed_digit, game_paused

    while True:
        data = safe_receive()
        if not data:
            time.sleep(0.01)
            continue

        current_time = time.time()

        if RECOGNITION_MODE == "direction":
            if data in DIRECTIONS:
                snake.change_direction(DIRECTIONS[data])
                hand_detected = True
                update_interaction_time()
            elif data == "pause":
                toggle_pause()

        elif RECOGNITION_MODE == "digit":
            if data in DIGITS:
                # 如果是新数字或数字变化，重置计时
                if data != current_digit:
                    current_digit = data
                    digit_start_time = current_time
                    confirmed_digit = None

                # 检查是否保持足够时间
                elif current_time - digit_start_time >= DIGIT_HOLD_TIME:
                    confirmed_digit = current_digit
                    last_digit = confirmed_digit  # 更新全局变量
                    update_interaction_time()
                    
                    # 处理暂停手势
                    if confirmed_digit == '9':
                        toggle_pause()
            else:
                # 如果没有识别到数字，重置状态
                current_digit = None
                digit_start_time = 0
                confirmed_digit = None

# ============================= 暂停功能 =============================
def toggle_pause():
    global game_paused, pause_start_time, pause_duration
    game_paused = not game_paused
    if game_paused:
        pause_start_time = time.time()
    else:
        pause_duration += time.time() - pause_start_time

def draw_pause_screen():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, 0))
    
    pause_text = font.render("PAUSED", True, WHITE)
    resume_text = small_font.render("Press P or show '9' to resume", True, WHITE)
    
    screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 50))
    screen.blit(resume_text, (WIDTH // 2 - resume_text.get_width() // 2, HEIGHT // 2 + 20))
    
    pygame.display.flip()

# ============================= 交互时间更新 =============================
def update_interaction_time():
    global last_interaction_time
    last_interaction_time = time.time()

def check_timeout():
    return time.time() - last_interaction_time > TIMEOUT

def draw_timeout_warning(screen, time_left):
    warning_text = small_font.render(f"Timeout in: {time_left:.0f}s", True, RED)
    screen.blit(warning_text, (WIDTH - warning_text.get_width() - 20, 20))

# ============================= 游戏界面函数 ================================
def main_menu():
    global RECOGNITION_MODE, hand_detected, last_digit, client, last_interaction_time

    # 切换到数字识别模式
    RECOGNITION_MODE = "digit"
    if not safe_send_with_retry("switch_to_digit"):
        print("警告: 无法切换到数字识别模式")
    last_digit = None
    update_interaction_time()

    # 启动控制线程
    dummy_snake = Snake()
    thread = threading.Thread(target=control_thread, args=(dummy_snake,), daemon=True)
    thread.start()

    while True:
        # 检查超时
        if check_timeout():
            return 0  # 超时退出游戏

        screen.fill(BLACK)
        title = font.render("Main Menu", True, WHITE)
        option1 = font.render("1. Single Player Game", True, WHITE)
        option2 = font.render("2. Human vs AI (Different Screen)", True, WHITE)
        option3 = font.render("3. Human vs AI (Same Screen)", True, WHITE)
        option4 = font.render("0. Exit Game", True, WHITE)

        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 4))
        screen.blit(option1, (WIDTH // 2 - option1.get_width() // 2, HEIGHT // 2))
        screen.blit(option2, (WIDTH // 2 - option2.get_width() // 2, HEIGHT // 2 + 60))
        screen.blit(option3, (WIDTH // 2 - option3.get_width() // 2, HEIGHT // 2 + 120))
        screen.blit(option4, (WIDTH // 2 - option4.get_width() // 2, HEIGHT // 2 + 180))

        if current_digit:
            draw_digit_confirmation(screen)
        
        # 显示手势图示
        if GESTURE_IMAGE:
            screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
        
        # 显示超时警告
        time_left = TIMEOUT - (time.time() - last_interaction_time)
        if time_left < 10:  # 最后10秒显示警告
            draw_timeout_warning(screen, time_left)

        pygame.display.flip()

        # 检测手势输入
        if confirmed_digit == '1':
            return 1
        elif confirmed_digit == '2':
            return 2
        elif confirmed_digit == '3':
            return 3
        elif confirmed_digit == '0':
            return 0

        # 检测键盘输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                update_interaction_time()
                if event.key == pygame.K_1:
                    return 1
                elif event.key == pygame.K_2:
                    return 2
                elif event.key == pygame.K_3:
                    return 3
                elif event.key == pygame.K_0:
                    return 0

        time.sleep(0.05)

def start_screen():
    global RECOGNITION_MODE, hand_detected, client, last_interaction_time

    RECOGNITION_MODE = "direction"
    if not safe_send_with_retry("switch_to_direction"):
        print("警告: 无法切换到方向识别模式")
    hand_detected = False  # 初始化为False
    update_interaction_time()

    # 启动方向识别线程（监听手部是否出现）
    dummy_snake = Snake()
    thread = threading.Thread(target=control_thread, args=(dummy_snake,), daemon=True)
    thread.start()

    waiting = True
    while waiting:
        # 检查超时
        if check_timeout():
            return False

        screen.fill(BLACK)
        title = font.render("Snake", True, WHITE)
        subtitle = font.render("show a hand to start", True, WHITE)

        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2))
        
        # 显示手势图示
        if GESTURE_IMAGE:
            screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
        
        # 显示超时警告
        time_left = TIMEOUT - (time.time() - last_interaction_time)
        if time_left < 10:  # 最后10秒显示警告
            draw_timeout_warning(screen, time_left)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                update_interaction_time()
                waiting = False

        # 检测手部出现，修改hand_detected为True
        if hand_detected:
            update_interaction_time()
            waiting = False

        time.sleep(0.05)

    for i in range(3, 0, -1):
        # 检查超时
        if check_timeout():
            return False

        screen.fill(BLACK)
        count_text = font.render(str(i), True, WHITE)
        screen.blit(count_text,
                    (WIDTH // 2 - count_text.get_width() // 2, HEIGHT // 2 - count_text.get_height() // 2))
        
        # 显示超时警告
        time_left = TIMEOUT - (time.time() - last_interaction_time)
        if time_left < 10:  # 最后10秒显示警告
            draw_timeout_warning(screen, time_left)

        pygame.display.flip()
        time.sleep(1)  # 每秒倒数一次

    # 复原hand_detected变量
    hand_detected = False
    return True

def game_over_screen(score):
    global RECOGNITION_MODE, last_digit, client, last_interaction_time

    # 切换到数字识别模式
    RECOGNITION_MODE = "digit"
    if not safe_send_with_retry("switch_to_digit"):
        print("警告: 无法切换到数字识别模式")
    last_digit = None
    update_interaction_time()

    # 显示界面
    while True:
        # 检查超时
        if check_timeout():
            return "menu"

        screen.fill(BLACK)
        msg1 = font.render(f"Game Over! Score: {score}", True, WHITE)
        msg2 = font.render("4. Back to Menu", True, WHITE)
        msg3 = font.render("5. Play Again", True, WHITE)
        msg4 = font.render("6. Replay it", True, WHITE)

        screen.blit(msg1, (WIDTH // 2 - msg1.get_width() // 2, HEIGHT // 3))
        screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2))
        screen.blit(msg3, (WIDTH // 2 - msg3.get_width() // 2, HEIGHT // 2 + 60))
        screen.blit(msg4, (WIDTH // 2 - msg4.get_width() // 2, HEIGHT // 2 + 120))

        draw_digit_confirmation(screen)
        
        # 显示手势图示
        if GESTURE_IMAGE:
            screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
        
        # 显示超时警告
        time_left = TIMEOUT - (time.time() - last_interaction_time)
        if time_left < 10:  # 最后10秒显示警告
            draw_timeout_warning(screen, time_left)

        pygame.display.flip()

        # 检测手势输入
        if confirmed_digit == '4':
            return "menu"
        elif confirmed_digit == '5':
            return "again"
        elif confirmed_digit == '6':
            return "replay"

        # 检测键盘输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                update_interaction_time()
                if event.key == pygame.K_4:
                    return "menu"
                elif event.key == pygame.K_5:
                    return "again"
                elif event.key == pygame.K_6:
                    return "replay"

        time.sleep(0.05)

def replay_game(replay_frames):
    global RECOGNITION_MODE, last_digit, current_digit, digit_start_time, confirmed_digit, client, last_interaction_time

    # 确保在数字识别模式
    RECOGNITION_MODE = "digit"
    if not safe_send_with_retry("switch_to_digit"):
        print("警告: 无法切换到数字识别模式")
    update_interaction_time()

    clock = pygame.time.Clock()

    while True:  # 主循环
        # 重置所有数字相关状态
        last_digit = None
        current_digit = None
        digit_start_time = 0
        confirmed_digit = None

        # 播放全部回放帧
        for frame in replay_frames:
            # 检查超时
            if check_timeout():
                return "menu"

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"

            screen.fill(BLACK)
            draw_border()
            draw_grid()

            temp_snake = Snake()
            temp_snake.body = [segment.copy() for segment in frame["snake_body"]]
            temp_snake.direction = frame["direction"]

            pygame.draw.circle(screen, RED, (frame["food"][0] + GRID_SIZE // 2, frame["food"][1] + GRID_SIZE // 2),
                               GRID_SIZE // 2)
            draw_snake(temp_snake)
            draw_score(frame["score"])
            draw_direction_keys(frame["direction"])
            
            # 显示超时警告
            time_left = TIMEOUT - (time.time() - last_interaction_time)
            if time_left < 10:  # 最后10秒显示警告
                draw_timeout_warning(screen, time_left)

            pygame.display.flip()
            clock.tick(12)

        # 回放结束后的提示
        waiting = True
        while waiting:  # 等待用户输入循环
            # 检查超时
            if check_timeout():
                return "menu"

            screen.fill(BLACK)
            game_over_text = font.render("Replay Finished! Show:", True, WHITE)
            option1 = font.render("4=Menu, 5=Play Again, 6=Replay", True, WHITE)
            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))
            screen.blit(option1, (WIDTH // 2 - option1.get_width() // 2, HEIGHT // 2))
            
            # 显示手势图示
            if GESTURE_IMAGE:
                screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
            
            # 显示超时警告
            time_left = TIMEOUT - (time.time() - last_interaction_time)
            if time_left < 10:  # 最后10秒显示警告
                draw_timeout_warning(screen, time_left)

            draw_digit_confirmation(screen)
            pygame.display.flip()

            # 手势输入检测
            if confirmed_digit == '4':
                return "menu"
            elif confirmed_digit == '5':
                return "again"
            elif confirmed_digit == '6':
                waiting = False  # 只退出内层循环，重新开始回放

            # 检查退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    update_interaction_time()
                    if event.key == pygame.K_4:
                        return "menu"
                    elif event.key == pygame.K_5:
                        return "again"
                    elif event.key == pygame.K_6:
                        waiting = False  # 重新开始回放

            time.sleep(0.05)

# ================================= 主游戏逻辑 ==================================
def option1_screen():
    global RECOGNITION_MODE, replay_frames, client, game_paused, last_interaction_time

    while True:
        if not start_screen():
            return False

        # 初始化游戏对象
        snake = Snake()
        food = generate_food(snake.body)
        score = 0
        temp_frames = []  # 临时存储当前游戏的帧数据
        game_paused = False
        pause_start_time = 0
        pause_duration = 0
        update_interaction_time()

        # 启动控制线程
        RECOGNITION_MODE = "direction"
        if not safe_send_with_retry("switch_to_direction"):
            print("警告: 无法切换到方向识别模式")
        threading.Thread(target=control_thread, args=(snake,), daemon=True).start()

        # 游戏主循环
        clock = pygame.time.Clock()
        last_move = pygame.time.get_ticks()
        running = True
        while running:
            # 检查超时
            if check_timeout():
                return True  # 返回主菜单

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    update_interaction_time()
                    if event.key == pygame.K_w:
                        snake.change_direction(UP)
                    elif event.key == pygame.K_s:
                        snake.change_direction(DOWN)
                    elif event.key == pygame.K_a:
                        snake.change_direction(LEFT)
                    elif event.key == pygame.K_d:
                        snake.change_direction(RIGHT)
                    elif event.key == pygame.K_p:
                        toggle_pause()

            # 处理暂停状态
            if game_paused:
                draw_pause_screen()
                clock.tick(8)
                continue

            # 蛇移动
            current_time = pygame.time.get_ticks()
            if current_time - last_move > 1000 // snake.speed:
                if not snake.move():
                    running = False
                last_move = current_time

            # 吃食物检测
            if snake.eat_food(food):
                food = generate_food(snake.body)
                score += 1
                snake.speed = min(10, 4 + score // 3)  # 速度随分数增加

            # 记录当前帧
            temp_frames.append({
                "snake_body": [segment.copy() for segment in snake.body],
                "food": food.copy(),
                "score": score,
                "direction": snake.direction
            })

            # 绘制
            screen.fill(BLACK)
            draw_border()
            draw_grid()
            draw_food(food)
            draw_snake(snake)
            draw_score(score)
            
            # 显示超时警告
            time_left = TIMEOUT - (time.time() - last_interaction_time)
            if time_left < 10:  # 最后10秒显示警告
                draw_timeout_warning(screen, time_left)

            pygame.display.flip()
            clock.tick(8)

        # 游戏结束，保存回放帧
        replay_frames = temp_frames.copy()
        result = game_over_screen(score)
        if result == "menu":
            return True
        elif result == "again":
            continue
        elif result == "replay":
            replay_result = replay_game(replay_frames)
            if replay_result == "menu":
                return True
            elif replay_result == "again":
                continue
            elif replay_result == "quit":
                return False

def option2_screen():
    global RECOGNITION_MODE, last_digit, current_digit, digit_start_time, confirmed_digit, client, last_interaction_time

    while True:
        # 启动 Human vs AI (Different Screen) 游戏
        pygame.display.set_mode(WINDOW_DUAL)  # 设置双屏窗口大小
        game_result = human_vs_ai_game()  # 假设返回 "menu" / "again" / "quit"
        update_interaction_time()

        # 游戏结束后切换到数字识别模式
        RECOGNITION_MODE = "digit"
        if not safe_send_with_retry("switch_to_digit"):
            print("警告: 无法切换到数字识别模式")

        # 重置数字识别状态
        last_digit = None
        current_digit = None
        digit_start_time = 0
        confirmed_digit = None

        # 显示游戏结束界面
        pygame.display.set_mode(WINDOW_SINGLE)  # 切换回单屏窗口大小
        waiting = True
        while waiting:
            # 检查超时
            if check_timeout():
                return True  # 返回主菜单

            screen.fill(BLACK)
            option_text = font.render("Show: 1=Menu, 2=Play Again", True, WHITE)
            screen.blit(option_text, (WIDTH // 2 - option_text.get_width() // 2, HEIGHT // 2))
            
            # 显示手势图示
            if GESTURE_IMAGE:
                screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
            
            # 显示超时警告
            time_left = TIMEOUT - (time.time() - last_interaction_time)
            if time_left < 10:  # 最后10秒显示警告
                draw_timeout_warning(screen, time_left)

            pygame.display.flip()

            # 检测手势输入
            if confirmed_digit == '1':
                return True  # 返回主菜单
            elif confirmed_digit == '2':
                waiting = False  # 重新开始游戏

            # 检测键盘输入
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # 完全退出游戏
                if event.type == pygame.KEYDOWN:
                    update_interaction_time()
                    if event.key == pygame.K_1:
                        return True  # 返回主菜单
                    elif event.key == pygame.K_2:
                        waiting = False  # 重新开始游戏

            time.sleep(0.05)

def option3_screen():
    global RECOGNITION_MODE, last_digit, current_digit, digit_start_time, confirmed_digit, client, last_interaction_time

    while True:
        # 启动 Human vs AI (Same Screen) 游戏
        pygame.display.set_mode(WINDOW_SINGLE)  # 确保窗口大小正确
        game_result = human_vs_ai_same_screen()  # 假设返回 "menu" / "again" / "quit"
        update_interaction_time()

        # 游戏结束后切换到数字识别模式
        RECOGNITION_MODE = "digit"
        if not safe_send_with_retry("switch_to_digit"):
            print("警告: 无法切换到数字识别模式")

        # 重置数字识别状态
        last_digit = None
        current_digit = None
        digit_start_time = 0
        confirmed_digit = None

        # 显示游戏结束界面
        screen.fill(BLACK)
        waiting = True
        while waiting:
            # 检查超时
            if check_timeout():
                return True  # 返回主菜单

            option_text = font.render("Show: 1=Menu, 2=Play Again", True, WHITE)
            screen.blit(option_text, (WIDTH // 2 - option_text.get_width() // 2, HEIGHT // 2))
            
            # 显示手势图示
            if GESTURE_IMAGE:
                screen.blit(GESTURE_IMAGE, GESTURE_IMAGE_RECT)
            
            # 显示超时警告
            time_left = TIMEOUT - (time.time() - last_interaction_time)
            if time_left < 10:  # 最后10秒显示警告
                draw_timeout_warning(screen, time_left)

            pygame.display.flip()

            # 等待手势或键盘输入
            # 检测手势输入
            if confirmed_digit == '1':
                return True  # 返回主菜单
            elif confirmed_digit == '2':
                waiting = False  # 重新开始游戏

            # 检测键盘输入
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # 完全退出游戏
                if event.type == pygame.KEYDOWN:
                    update_interaction_time()
                    if event.key == pygame.K_1:
                        return True  # 返回主菜单
                    elif event.key == pygame.K_2:
                        waiting = False  # 重新开始游戏

            time.sleep(0.05)

# ================================== 辅助函数 ===================================
def generate_food(snake_body):
    while True:
        # 99.9% 正常刷新
        if random.random() > 0.001:  # 99.9%的概率
            x = random.randint(1, GRID_COUNT - 2) * GRID_SIZE + BORDER_WIDTH
            y = random.randint(1, GRID_COUNT - 2) * GRID_SIZE + BORDER_WIDTH
        else:  # 0.1%的概率可能刷在边缘
            x = random.choice([0, GRID_COUNT - 1]) * GRID_SIZE + BORDER_WIDTH
            y = random.choice([0, GRID_COUNT - 1]) * GRID_SIZE + BORDER_WIDTH

        pos = [x, y]
        if pos not in snake_body:
            return pos

def draw_border():
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (0, HEIGHT - BORDER_WIDTH, WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, BORDER_WIDTH, HEIGHT))
    pygame.draw.rect(screen, BORDER_COLOR, (WIDTH - BORDER_WIDTH, 0, BORDER_WIDTH, HEIGHT))

def draw_grid():
    for x in range(BORDER_WIDTH, WIDTH - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, BORDER_WIDTH), (x, HEIGHT - BORDER_WIDTH))
    for y in range(BORDER_WIDTH, HEIGHT - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (BORDER_WIDTH, y), (WIDTH - BORDER_WIDTH, y))

def draw_food(pos):
    pygame.draw.circle(screen, RED, (pos[0] + GRID_SIZE // 2, pos[1] + GRID_SIZE // 2), GRID_SIZE // 2)

def draw_snake(snake):
    for i, seg in enumerate(snake.body):
        color = (0, max(50, 255 - i * 20), 0)
        pygame.draw.rect(screen, color, (*seg, GRID_SIZE, GRID_SIZE))

        # 绘制蛇头眼睛
        if i == 0:
            eye_pos1 = (seg[0] + GRID_SIZE // 3, seg[1] + GRID_SIZE // 3)
            eye_pos2 = (seg[0] + GRID_SIZE * 2 // 3, seg[1] + GRID_SIZE // 3)
            pygame.draw.circle(screen, WHITE, eye_pos1, GRID_SIZE // 8)
            pygame.draw.circle(screen, WHITE, eye_pos2, GRID_SIZE // 8)

def draw_score(score):
    text = font.render(f"score: {score}", True, WHITE)
    screen.blit(text, (BORDER_WIDTH + 10, BORDER_WIDTH // 2 - 10))

def draw_digit_confirmation(screen):
    global current_digit, digit_start_time, confirmed_digit

    if current_digit:
        # 计算当前数字的保持时间
        hold_time = time.time() - digit_start_time
        progress = min(1.0, hold_time / DIGIT_HOLD_TIME)

        # 绘制进度条背景
        bar_width = 200
        bar_height = 20
        bar_x = WIDTH // 2 - bar_width // 2
        bar_y = HEIGHT - 100
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_width, bar_height))

        # 动态颜色：从红色渐变到绿色
        color_progress = min(1.0, hold_time / DIGIT_HOLD_TIME)
        r = int(255 * (1 - color_progress))
        g = int(255 * color_progress)
        progress_color = (r, g, 0)

        # 绘制进度条
        pygame.draw.rect(screen, progress_color,
                         (bar_x, bar_y, int(bar_width * progress), bar_height))

        # 显示当前识别的数字和倒计时
        if hold_time < DIGIT_HOLD_TIME:
            remaining = DIGIT_HOLD_TIME - hold_time
            digit_text = font.render(
                f"Hold {current_digit}: {remaining:.1f}s", True, WHITE)
        else:
            digit_text = font.render(
                f"Confirmed: {current_digit}!", True, GREEN)
            confirmed_digit = current_digit

        screen.blit(digit_text, (WIDTH // 2 - digit_text.get_width() // 2, bar_y - 40))

        # 绘制边框增加视觉效果
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
    else:
        # 没有检测到数字时的提示
        prompt_text = font.render("Show a number (0-9)", True, GRAY)
        screen.blit(prompt_text, (WIDTH // 2 - prompt_text.get_width() // 2, HEIGHT - 140))

def draw_direction_keys(current_direction):
    key_positions = {
        UP: (WIDTH // 2 - 15, BORDER_WIDTH // 2 - 15),
        DOWN: (WIDTH // 2 - 15, HEIGHT - BORDER_WIDTH // 2 - 15),
        LEFT: (BORDER_WIDTH // 2 - 15, HEIGHT // 2 - 15),
        RIGHT: (WIDTH - BORDER_WIDTH // 2 - 15, HEIGHT // 2 - 15)
    }

    key_labels = {
        UP: "W",
        DOWN: "D",
        LEFT: "A",
        RIGHT: "S"
    }

    for direction, pos in key_positions.items():
        color = GREEN if direction == current_direction else GRAY
        key_text = font.render(key_labels[direction], True, color)
        screen.blit(key_text, pos)


# =================================== 主程序 ===================================
if __name__ == "__main__":
    try:
        client = init_connection()  # 初始化socket连接
        update_interaction_time()  # 初始化交互时间

        while True:
            # 显示主菜单
            choice = main_menu()

            if choice == 1:
                # 开始单人游戏
                should_continue1 = option1_screen()
                if not should_continue1:
                    break
            elif choice == 2:
                # 跳转到Human vs AI (Different Screen)
                should_continue2 = option2_screen()
                if not should_continue2:
                    break
            elif choice == 3:
                # 跳转到Human vs AI (Same Screen)
                should_continue3 = option3_screen()
                if not should_continue3:
                    break
            elif choice == 0:
                break
    finally:
        pygame.quit()
        close_socket()
        print("游戏已退出")