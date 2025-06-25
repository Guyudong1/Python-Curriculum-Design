import pygame
import sys
from Human_Vs_AI_DIFScreen import main_game as human_vs_ai_game
from Human_VS_AI_SameScreen import main_game as human_vs_ai_same_screen

# ================================== 全局变量 ==================================
# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)

# 初始化pygame
pygame.init()
font = pygame.font.SysFont(None, 48)


# =============================== 游戏界面函数 ================================
def main_menu():
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("贪吃蛇游戏菜单")

    while True:
        screen.fill(BLACK)
        title = font.render("Snake Game Menu", True, WHITE)
        option1 = font.render("1. Human vs AI (Different Screen)", True, WHITE)
        option2 = font.render("2. Human vs AI (Same Screen)", True, WHITE)
        option3 = font.render("0. Exit Game", True, WHITE)

        screen.blit(title, (400 - title.get_width() // 2, 100))
        screen.blit(option1, (400 - option1.get_width() // 2, 250))
        screen.blit(option2, (400 - option2.get_width() // 2, 350))
        screen.blit(option3, (400 - option3.get_width() // 2, 450))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 1
                elif event.key == pygame.K_2:
                    return 2
                elif event.key == pygame.K_0:
                    return 0


def option1_screen():
    while True:
        again = human_vs_ai_game()
        if not again:
            return True  # 返回主菜单


def option2_screen():
    while True:
        again = human_vs_ai_same_screen()
        if not again:
            return True  # 返回主菜单


# =================================== 主程序 ===================================
if __name__ == "__main__":
    try:
        pygame.display.set_mode(WINDOW_SINGLE)  # 初始设置为单屏模式

        while True:
            # 显示主菜单
            choice = main_menu()

            if choice == 1:
                # 单人游戏模式
                while True:
                    should_continue = option1_screen()
                    if should_continue == "menu":
                        break  # 返回主菜单
                    elif should_continue == False:
                        sys.exit()  # 完全退出游戏

            elif choice == 2:
                # Human vs AI (Different Screen)
                while True:
                    current_mode = pygame.display.get_surface().get_size()
                    try:
                        pygame.display.set_mode(WINDOW_DUAL)
                        should_continue = option2_screen()
                        if should_continue == "menu":
                            pygame.display.set_mode(WINDOW_SINGLE)
                            break  # 返回主菜单
                        elif should_continue == False:
                            sys.exit()  # 完全退出游戏
                    finally:
                        pygame.display.set_mode(current_mode)
                        close_socket()
                        client = None

            elif choice == 3:
                # Human vs AI (Same Screen)
                while True:
                    should_continue = option3_screen()
                    if should_continue == "menu":
                        break  # 返回主菜单
                    elif should_continue == False:
                        sys.exit()  # 完全退出游戏

            elif choice == 0:
                break  # 退出游戏

    finally:
        pygame.quit()
        close_socket()
        print("游戏已退出")