import pygame
import random

# 初始化pygame
pygame.init()
GRID_SIZE = 50
GRID_COUNT = 12
BORDER_WIDTH = 80
WIDTH, HEIGHT = GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2, GRID_SIZE * GRID_COUNT + BORDER_WIDTH * 2

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

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇游戏")

font = pygame.font.SysFont(None, 48)

class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [
            [BORDER_WIDTH + GRID_SIZE * 2, BORDER_WIDTH + GRID_SIZE * 2],
            [BORDER_WIDTH + GRID_SIZE, BORDER_WIDTH + GRID_SIZE * 2],
            [BORDER_WIDTH, BORDER_WIDTH + GRID_SIZE * 2]
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

def generate_food(snake_body):
    while True:
        pos = [
            random.randint(0, GRID_COUNT - 1) * GRID_SIZE + BORDER_WIDTH,
            random.randint(0, GRID_COUNT - 1) * GRID_SIZE + BORDER_WIDTH
        ]
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

def start_screen():
    screen.fill(BLACK)
    title = font.render("贪吃蛇游戏", True, WHITE)
    subtitle = font.render("Press any key to start", True, WHITE)

    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2))
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

def game_over_screen(score):
    screen.fill(BLACK)
    msg1 = font.render(f"Game Over! Score: {score}", True, WHITE)
    msg2 = font.render("1 to Exit, 2 to Restart", True, WHITE)

    screen.blit(msg1, (WIDTH // 2 - msg1.get_width() // 2, HEIGHT // 3))
    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return False
                elif event.key == pygame.K_2:
                    return True
        pygame.time.delay(50)

def main_game():
    if not start_screen():
        return False

    # 初始化游戏对象
    snake = Snake()
    food = generate_food(snake.body)
    score = 0

    # 游戏主循环
    clock = pygame.time.Clock()
    last_move = pygame.time.get_ticks()
    running = True

    while running:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction(RIGHT)

        # 蛇移动
        if pygame.time.get_ticks() - last_move > 1000 // snake.speed:
            done = not snake.move()
            last_move = pygame.time.get_ticks()

            # 吃食物检测
            ate_food = snake.eat_food(food)
            if ate_food:
                food = generate_food(snake.body)
                score += 1
                snake.speed = min(10, 4 + score // 3)

            if done:
                running = False

        # 绘制
        screen.fill(BLACK)
        draw_border()
        draw_grid()
        draw_food(food)
        draw_snake(snake)
        draw_score(score)
        pygame.display.flip()
        clock.tick(60)

    return game_over_screen(score)

if __name__ == "__main__":
    try:
        while True:
            if not main_game():
                break
    finally:
        pygame.quit()
        print("游戏已退出")