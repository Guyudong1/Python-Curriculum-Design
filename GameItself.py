import pygame
import random
import time
from Human_Vs_AI_DIFScreen import main_game as human_vs_ai_game
from Human_VS_AI_SameScreen import main_game as human_vs_ai_same_screen

# 初始化pygame
pygame.init()
# 全局变量，记录方向操作,用于存储每帧画面数据
global replay_frames
replay_frames = []  # 每次新游戏前清空画面记录
# 屏幕大小
GRID_SIZE = 50
GRID_COUNT = 12
BORDER_WIDTH = 80  # 增加边缘宽度
WIDTH, HEIGHT = GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2, GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2
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
# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇游戏")
# 字体
font = pygame.font.SysFont(None, 26)
class Snake:
    def __init__(self):
        self.body = [[BORDER_WIDTH + GRID_SIZE * 2, BORDER_WIDTH + GRID_SIZE * 2],
                     [BORDER_WIDTH + GRID_SIZE, BORDER_WIDTH + GRID_SIZE * 2],
                     [BORDER_WIDTH, BORDER_WIDTH + GRID_SIZE * 2]]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.grow = False
        self.collision_frames = 0

    def move(self):
        self.direction = self.next_direction
        head = [self.body[0][0] + self.direction[0] * GRID_SIZE,
                self.body[0][1] + self.direction[1] * GRID_SIZE]

        if head in self.body or head[0] < BORDER_WIDTH or head[0] >= WIDTH - BORDER_WIDTH or head[1] < BORDER_WIDTH or \
                head[1] >= HEIGHT - BORDER_WIDTH:
            self.collision_frames += 1
            if self.collision_frames >= 1:
                return False
        else:
            self.collision_frames = 0

        self.body.insert(0, head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        return True

    def change_direction(self, new_direction):
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.next_direction = new_direction

    def eat_food(self, food_pos):
        if self.body[0] == food_pos:
            self.grow = True
            return True
        return False
def draw_grid():
    for x in range(BORDER_WIDTH, WIDTH - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, BORDER_WIDTH), (x, HEIGHT - BORDER_WIDTH))
    for y in range(BORDER_WIDTH, HEIGHT - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (BORDER_WIDTH, y), (WIDTH - BORDER_WIDTH, y))
def draw_border():
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (0, HEIGHT - BORDER_WIDTH, WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, BORDER_WIDTH, HEIGHT))
    pygame.draw.rect(screen, BORDER_COLOR, (WIDTH - BORDER_WIDTH, 0, BORDER_WIDTH, HEIGHT))
def generate_food(snake_body):
    while True:
        x = random.randint(0, GRID_COUNT - 1)
        y = random.randint(0, GRID_COUNT - 1)
        food_pos = [(x * GRID_SIZE) + BORDER_WIDTH, (y * GRID_SIZE) + BORDER_WIDTH]
        if food_pos not in snake_body:
            return food_pos
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
def draw_snake(snake):
    head_color = DARK_GREEN  # 头部颜色（最深）
    tail_color = GREEN  # 尾部颜色（最浅）

    body_length = len(snake.body)

    for i, segment in enumerate(snake.body):
        # 计算颜色渐变（从头到尾）
        color = pygame.Color.lerp(pygame.Color(head_color), pygame.Color(tail_color), i / max(1, body_length - 1))
        pygame.draw.rect(screen, color, (*segment, GRID_SIZE, GRID_SIZE))

        if i == 0:  # 绘制蛇眼睛
            eye_size = GRID_SIZE // 6
            eye_offset_x = GRID_SIZE // 4
            eye_offset_y = GRID_SIZE // 4

            eye1 = (segment[0] + eye_offset_x, segment[1] + eye_offset_y)
            eye2 = (segment[0] + GRID_SIZE - eye_offset_x, segment[1] + eye_offset_y)

            pygame.draw.circle(screen, WHITE, eye1, eye_size)
            pygame.draw.circle(screen, WHITE, eye2, eye_size)
def draw_score(score):
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (BORDER_WIDTH + 10, HEIGHT - BORDER_WIDTH + 10))
def main_menu():
    screen.fill(BLACK)
    menu_font = pygame.font.SysFont(None, 48)
    lines = [
        "Snake",
        "1. Single Game",
        "2. Human vs AI dif",
        "3. Human vs AI same",
        "4. Exit"
    ]
    for idx, txt in enumerate(lines):
        menu_text = menu_font.render(txt, True, WHITE)
        text_x = WIDTH // 2 - menu_text.get_width() // 2
        text_y = 120 + idx * 60
        screen.blit(menu_text, (text_x, text_y))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "exit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "single"
                elif event.key == pygame.K_2:
                    return "human_ai_dif"
                elif event.key == pygame.K_3:
                    return "human_ai_same"
                elif event.key == pygame.K_4:
                    return "exit"
        pygame.time.delay(50)
def game_over_screen(snake, food, score):
    game_over_text = font.render("Game Over!Press 1=Return, 2=Repaly", True, WHITE)
    text_x = WIDTH // 2 - game_over_text.get_width() // 2
    text_y = BORDER_WIDTH // 2 - 10
    screen.blit(game_over_text, (text_x, text_y))

    pygame.draw.circle(screen, RED, (food[0] + GRID_SIZE // 2, food[1] + GRID_SIZE // 2), GRID_SIZE // 2)
    draw_snake(snake)
    draw_score(score)
    draw_direction_keys(snake.direction)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "exit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "restart"
                elif event.key == pygame.K_2:
                    return "replay"
def Run_SnakeGame():
    global replay_frames
    while True:
        menu_choice = main_menu()
        if menu_choice == "exit":
            break
        elif menu_choice == "single":
            # 单人模式
            temp_frames = []
            clock = pygame.time.Clock()
            snake = Snake()
            food = generate_food(snake.body)
            score = 0
            running = True

            while running:
                screen.fill(BLACK)
                draw_border()
                draw_grid()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            snake.change_direction(UP)
                        elif event.key == pygame.K_DOWN:
                            snake.change_direction(DOWN)
                        elif event.key == pygame.K_LEFT:
                            snake.change_direction(LEFT)
                        elif event.key == pygame.K_RIGHT:
                            snake.change_direction(RIGHT)
                if not snake.move():
                    break
                if snake.eat_food(food):
                    food = generate_food(snake.body)
                    score += 1
                pygame.draw.circle(screen, RED, (food[0] + GRID_SIZE // 2, food[1] + GRID_SIZE // 2), GRID_SIZE // 2)
                draw_snake(snake)
                draw_score(score)
                draw_direction_keys(snake.direction)
                pygame.display.flip()
                clock.tick(8)
                temp_frames.append({
                    "snake_body": [segment.copy() for segment in snake.body],
                    "food": food.copy(),
                    "score": score,
                    "direction": snake.direction
                })

            replay_frames = temp_frames.copy()
            result = game_over_screen(snake, food, score)
            if result == "exit":
                break
            elif result == "replay":
                next_action = replay_game(replay_frames)
                if next_action == "exit":
                    break

        elif menu_choice == "human_ai_dif":
            pygame.display.set_mode(WINDOW_DUAL)
            while True:
                again = human_vs_ai_game()
                if not again:
                    break
            pygame.display.set_mode(WINDOW_SINGLE)
        elif menu_choice == "human_ai_same":
            while True:
                again = human_vs_ai_same_screen()
                if not again:
                    break
def replay_game(replay_frames):
    clock = pygame.time.Clock()
    while True:  # 外层循环用于支持“2=Replay”功能
        # 播放全部回放帧
        for frame in replay_frames:
            screen.fill(BLACK)
            draw_border()
            draw_grid()

            temp_snake = Snake()
            temp_snake.body = [segment.copy() for segment in frame["snake_body"]]
            temp_snake.direction = frame["direction"]

            pygame.draw.circle(screen, RED, (frame["food"][0] + GRID_SIZE // 2, frame["food"][1] + GRID_SIZE // 2), GRID_SIZE // 2)
            draw_snake(temp_snake)
            draw_score(frame["score"])
            draw_direction_keys(frame["direction"])

            pygame.display.flip()
            clock.tick(8)

        # 回放结束后显示提示并等待按键
        game_over_text = font.render("Game Over! Press 1=Return, 2=Replay", True, WHITE)
        text_x = WIDTH // 2 - game_over_text.get_width() // 2
        text_y = BORDER_WIDTH // 2 - 10
        screen.blit(game_over_text, (text_x, text_y))
        pygame.display.flip()

        # 等待用户输入
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        return "restart"   # 回主菜单
                    elif event.key == pygame.K_2:
                        break           # 跳出等待，进入外层while重播
            else:
                pygame.time.delay(50)
                continue
            break  # 仅 break if "2" pressed


if __name__ == "__main__":
    Run_SnakeGame()
    pygame.quit()
    exit()

