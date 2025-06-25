import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 初始化pygame
pygame.init()

# 游戏参数
GRID_SIZE = 40  # 缩小网格大小以适应双屏
GRID_COUNT = 12
BORDER_WIDTH = 60
SCREEN_WIDTH = GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2
SCREEN_HEIGHT = GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2

# 双屏总宽度
TOTAL_WIDTH = SCREEN_WIDTH * 2
HEIGHT = SCREEN_HEIGHT

# 颜色定义
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
DARK_GREEN = (0, 90, 0)
DARK_BLUE = (0, 0, 90)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BORDER_COLOR = (50, 50, 50)
DIVIDER_COLOR = (150, 150, 150)

# 方向定义
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
pygame.display.set_caption("Human vs AI Snake Battle - Dual Screen")

font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 28)


class Snake:
    def __init__(self, is_ai=False, offset_x=0):
        self.is_ai = is_ai
        self.offset_x = offset_x  # 屏幕偏移量
        self.reset()

    def reset(self):
        self.body = [
            [self.offset_x + BORDER_WIDTH + GRID_SIZE * 2, BORDER_WIDTH + GRID_SIZE * 2],
            [self.offset_x + BORDER_WIDTH + GRID_SIZE, BORDER_WIDTH + GRID_SIZE * 2],
            [self.offset_x + BORDER_WIDTH, BORDER_WIDTH + GRID_SIZE * 2]
        ]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.grow = False
        self.speed = 4
        self.alive = True
        self.score = 0

    def move(self):
        if not self.alive:
            return False

        self.direction = self.next_direction
        head = [
            self.body[0][0] + self.direction[0] * GRID_SIZE,
            self.body[0][1] + self.direction[1] * GRID_SIZE
        ]

        # 墙碰撞
        if (head[0] < self.offset_x + BORDER_WIDTH or
                head[0] >= self.offset_x + SCREEN_WIDTH - BORDER_WIDTH or
                head[1] < BORDER_WIDTH or
                head[1] >= HEIGHT - BORDER_WIDTH):
            self.alive = False
            return False

        # 身体自撞判定
        if head in self.body:
            self.alive = False
            return False

        self.body.insert(0, head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
            self.score += 1
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


def generate_food(snake_body, offset_x):
    while True:
        pos = [
            offset_x + BORDER_WIDTH + random.randint(0, GRID_COUNT - 1) * GRID_SIZE,
            BORDER_WIDTH + random.randint(0, GRID_COUNT - 1) * GRID_SIZE
        ]
        if pos not in snake_body:
            return pos


def draw_border(offset_x):
    # 绘制游戏区域边框
    pygame.draw.rect(screen, BORDER_COLOR, (offset_x, 0, SCREEN_WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (offset_x, HEIGHT - BORDER_WIDTH, SCREEN_WIDTH, BORDER_WIDTH))
    pygame.draw.rect(screen, BORDER_COLOR, (offset_x, 0, BORDER_WIDTH, HEIGHT))
    pygame.draw.rect(screen, BORDER_COLOR, (offset_x + SCREEN_WIDTH - BORDER_WIDTH, 0, BORDER_WIDTH, HEIGHT))


def draw_grid(offset_x):
    # 绘制网格
    for x in range(offset_x + BORDER_WIDTH, offset_x + SCREEN_WIDTH - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, BORDER_WIDTH), (x, HEIGHT - BORDER_WIDTH))
    for y in range(BORDER_WIDTH, HEIGHT - BORDER_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (offset_x + BORDER_WIDTH, y), (offset_x + SCREEN_WIDTH - BORDER_WIDTH, y))


def draw_food(pos):
    pygame.draw.circle(screen, RED, (pos[0] + GRID_SIZE // 2, pos[1] + GRID_SIZE // 2), GRID_SIZE // 2)


def draw_snake(snake):
    for i, seg in enumerate(snake.body):
        # 根据蛇的类型选择颜色
        if snake.is_ai:
            color = (0, 0, max(50, 255 - i * 20))  # AI蛇用蓝色
        else:
            color = (0, max(50, 255 - i * 20), 0)  # 人类蛇用绿色

        # 绘制蛇身
        pygame.draw.rect(screen, color, (seg[0], seg[1], GRID_SIZE, GRID_SIZE))

        # 绘制蛇头眼睛
        if i == 0:
            eye_pos1 = (seg[0] + GRID_SIZE // 3, seg[1] + GRID_SIZE // 3)
            eye_pos2 = (seg[0] + GRID_SIZE * 2 // 3, seg[1] + GRID_SIZE // 3)
            pygame.draw.circle(screen, WHITE, eye_pos1, GRID_SIZE // 8)
            pygame.draw.circle(screen, WHITE, eye_pos2, GRID_SIZE // 8)


def draw_scores(human_score, ai_score):
    # 人类分数(左屏)
    human_text = small_font.render(f"Human: {human_score}", True, GREEN)
    screen.blit(human_text, (BORDER_WIDTH + 10, BORDER_WIDTH // 2 - 10))

    # AI分数(右屏)
    ai_text = small_font.render(f"AI: {ai_score}", True, BLUE)
    screen.blit(ai_text, (SCREEN_WIDTH + BORDER_WIDTH + 10, BORDER_WIDTH // 2 - 10))


def draw_divider():
    # 绘制屏幕分隔线
    pygame.draw.line(screen, DIVIDER_COLOR, (SCREEN_WIDTH, 0), (SCREEN_WIDTH, HEIGHT), 3)


def start_screen():
    screen.fill(BLACK)
    title = font.render("Human vs AI Snake Battle", True, WHITE)
    subtitle = font.render("Press any key to start", True, WHITE)
    controls = small_font.render("Arrow keys control the green snake (left screen)", True, GREEN)

    screen.blit(title, (TOTAL_WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
    screen.blit(subtitle, (TOTAL_WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2))
    screen.blit(controls, (TOTAL_WIDTH // 2 - controls.get_width() // 2, HEIGHT * 2 // 3))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.delay(50)

    return True


def game_over_screen(human_score, ai_score):
    screen.fill(BLACK)

    if human_score > ai_score:
        result = font.render("Human wins!", True, GREEN)
    elif human_score < ai_score:
        result = font.render("AI wins!", True, BLUE)
    else:
        result = font.render("It's a draw!", True, WHITE)

    score_text = font.render(f"Final Score: {human_score} - {ai_score}", True, WHITE)
    msg = font.render("1.Return, 2.Play again", True, WHITE)

    screen.blit(result, (TOTAL_WIDTH // 2 - result.get_width() // 2, HEIGHT // 3))
    screen.blit(score_text, (TOTAL_WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
    screen.blit(msg, (TOTAL_WIDTH // 2 - msg.get_width() // 2, HEIGHT * 2 // 3))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "exit"
                if event.key == pygame.K_2:
                    return "human_ai_dif"
        pygame.time.delay(50)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载AI模型
model = DQN(151, 4)
model.load_state_dict(torch.load(r"E:\keshe\Python\QL_snake\pkl\snake_dqn.pth"))
model.eval()


def get_state(snake, food, offset_x):
    # Create a grid representation of the game state
    grid = np.zeros((GRID_COUNT, GRID_COUNT))

    # Mark snake body (except head) as -1
    for segment in snake.body[1:]:
        x = (segment[0] - offset_x - BORDER_WIDTH) // GRID_SIZE
        y = (segment[1] - BORDER_WIDTH) // GRID_SIZE
        if 0 <= x < GRID_COUNT and 0 <= y < GRID_COUNT:
            grid[y, x] = -1

    # Mark snake head as 1
    head_x = (snake.body[0][0] - offset_x - BORDER_WIDTH) // GRID_SIZE
    head_y = (snake.body[0][1] - BORDER_WIDTH) // GRID_SIZE
    if 0 <= head_x < GRID_COUNT and 0 <= head_y < GRID_COUNT:
        grid[head_y, head_x] = 1

    # Mark food position as 2
    food_x = (food[0] - offset_x - BORDER_WIDTH) // GRID_SIZE
    food_y = (food[1] - BORDER_WIDTH) // GRID_SIZE
    if 0 <= food_x < GRID_COUNT and 0 <= food_y < GRID_COUNT:
        grid[food_y, food_x] = 2

    # Add danger signals (distance to walls and body)
    head = snake.body[0]
    danger_straight = 0
    danger_left = 0
    danger_right = 0

    # Directions relative to current direction
    if snake.direction == (0, -1):  # UP
        directions = {
            'left': (-1, 0),
            'right': (1, 0),
            'straight': (0, -1)
        }
    elif snake.direction == (0, 1):  # DOWN
        directions = {
            'left': (1, 0),
            'right': (-1, 0),
            'straight': (0, 1)
        }
    elif snake.direction == (-1, 0):  # LEFT
        directions = {
            'left': (0, 1),
            'right': (0, -1),
            'straight': (-1, 0)
        }
    else:  # RIGHT
        directions = {
            'left': (0, -1),
            'right': (0, 1),
            'straight': (1, 0)
        }

    # Check for danger in each direction
    for direction_name, direction in directions.items():
        next_pos = (head[0] + direction[0] * GRID_SIZE, head[1] + direction[1] * GRID_SIZE)

        # Check if next position is out of bounds or in snake body
        if (next_pos[0] < offset_x + BORDER_WIDTH or
                next_pos[0] >= offset_x + SCREEN_WIDTH - BORDER_WIDTH or
                next_pos[1] < BORDER_WIDTH or
                next_pos[1] >= HEIGHT - BORDER_WIDTH or
                list(next_pos) in snake.body[1:]):
            if direction_name == 'straight':
                danger_straight = 1
            elif direction_name == 'left':
                danger_left = 1
            elif direction_name == 'right':
                danger_right = 1

    # Calculate food direction
    food_dir_x = 0
    food_dir_y = 0
    if food_x < head_x:
        food_dir_x = -1
    elif food_x > head_x:
        food_dir_x = 1

    if food_y < head_y:
        food_dir_y = -1
    elif food_y > head_y:
        food_dir_y = 1

    # Flatten the grid and add additional features
    state = grid.flatten().tolist()
    state += [danger_straight, danger_left, danger_right]
    state += [food_dir_x, food_dir_y]
    state += list(snake.direction)

    return np.array(state, dtype=np.float32)


def main_game():
    if not start_screen():
        return False

    # 初始化游戏对象
    human_snake = Snake(is_ai=False, offset_x=0)  # 左屏
    ai_snake = Snake(is_ai=True, offset_x=SCREEN_WIDTH)  # 右屏
    human_food = generate_food(human_snake.body, 0)
    ai_food = generate_food(ai_snake.body, SCREEN_WIDTH)

    # 游戏主循环
    clock = pygame.time.Clock()
    last_human_move = pygame.time.get_ticks()
    last_ai_move = pygame.time.get_ticks()
    running = True

    while running:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    human_snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    human_snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    human_snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    human_snake.change_direction(RIGHT)

        # 人类蛇移动(左屏)
        if pygame.time.get_ticks() - last_human_move > 1000 // human_snake.speed:
            if human_snake.alive:
                human_snake.move()

                # 吃食物检测
                ate_food = human_snake.eat_food(human_food)
                if ate_food:
                    human_food = generate_food(human_snake.body, 0)
                    human_snake.speed = min(10, 4 + human_snake.score // 3)

            last_human_move = pygame.time.get_ticks()

        # AI蛇移动(右屏)
        if pygame.time.get_ticks() - last_ai_move > 1000 // ai_snake.speed:
            if ai_snake.alive:
                # 获取AI状态并决定动作
                state = get_state(ai_snake, ai_food, SCREEN_WIDTH)
                state = torch.from_numpy(state).float().unsqueeze(0)

                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

                # 根据动作改变AI蛇的方向
                if action == 0:
                    ai_snake.change_direction(UP)
                elif action == 1:
                    ai_snake.change_direction(DOWN)
                elif action == 2:
                    ai_snake.change_direction(LEFT)
                elif action == 3:
                    ai_snake.change_direction(RIGHT)

                ai_snake.move()

                # 吃食物检测
                ate_food = ai_snake.eat_food(ai_food)
                if ate_food:
                    ai_food = generate_food(ai_snake.body, SCREEN_WIDTH)
                    ai_snake.speed = min(10, 4 + ai_snake.score // 3)

            last_ai_move = pygame.time.get_ticks()

        # 检查游戏是否结束(双方都死亡)
        if not human_snake.alive and not ai_snake.alive:
            running = False
            # 人类死了，AI活着，AI分数领先，立即结束
        elif not human_snake.alive and ai_snake.alive:
            if ai_snake.score > human_snake.score:
                running = False
            # AI死了，人类活着，人类分数领先，立即结束
        elif not ai_snake.alive and human_snake.alive:
            if human_snake.score > ai_snake.score:
                running = False

        # 绘制
        screen.fill(BLACK)

        # 绘制左屏(人类)
        draw_border(0)
        draw_grid(0)
        draw_food(human_food)
        draw_snake(human_snake)

        # 绘制右屏(AI)
        draw_border(SCREEN_WIDTH)
        draw_grid(SCREEN_WIDTH)
        draw_food(ai_food)
        draw_snake(ai_snake)

        # 绘制分隔线和分数
        draw_divider()
        draw_scores(human_snake.score, ai_snake.score)

        pygame.display.flip()
        clock.tick(144)

    while True:
        result = game_over_screen(human_snake.score, ai_snake.score)
        if result == "restart" or result == "human_ai_dif":
            # 重新开始本地 main_game 循环
            return True
        else:
            return False


# 主程序入口
if __name__ == "__main__":
    try:
        while True:
            if not main_game():
                break
    finally:
        pygame.quit()
        print("游戏已退出")