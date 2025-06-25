import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import pickle
from init_Snake import Snake, generate_food, GRID_COUNT, BORDER_WIDTH, GRID_SIZE, WIDTH, HEIGHT

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def get_state(snake, food):
    # Create a grid representation of the game state
    grid = np.zeros((GRID_COUNT, GRID_COUNT))
    
    # Mark snake body (except head) as -1
    for segment in snake.body[1:]:
        x = (segment[0] - BORDER_WIDTH) // GRID_SIZE
        y = (segment[1] - BORDER_WIDTH) // GRID_SIZE
        if 0 <= x < GRID_COUNT and 0 <= y < GRID_COUNT:
            grid[y, x] = -1
    
    # Mark snake head as 1
    head_x = (snake.body[0][0] - BORDER_WIDTH) // GRID_SIZE
    head_y = (snake.body[0][1] - BORDER_WIDTH) // GRID_SIZE
    if 0 <= head_x < GRID_COUNT and 0 <= head_y < GRID_COUNT:
        grid[head_y, head_x] = 1
    
    # Mark food position as 2
    food_x = (food[0] - BORDER_WIDTH) // GRID_SIZE
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
        if (next_pos[0] < BORDER_WIDTH or next_pos[0] >= WIDTH - BORDER_WIDTH or
            next_pos[1] < BORDER_WIDTH or next_pos[1] >= HEIGHT - BORDER_WIDTH or
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

def train_dqn(episodes=1000, batch_size=32, save_path="E:/keshe/Python/QL_snake/pkl/snake_dqn.pth"):
    # Initialize environment and agent
    snake = Snake()
    food = generate_food(snake.body)
    
    # State size: grid (12x12=144) + danger(3) + food_dir(2) + snake_dir(2) = 151
    state_size = GRID_COUNT * GRID_COUNT + 3 + 2 + 2
    action_size = 4  # UP, DOWN, LEFT, RIGHT
    
    agent = DQNAgent(state_size, action_size)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for e in range(episodes):
        snake.reset()
        food = generate_food(snake.body)
        state = get_state(snake, food)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get action
            action = agent.act(state)
            
            # Map action to direction
            if action == 0:
                new_dir = (0, -1)  # UP
            elif action == 1:
                new_dir = (0, 1)   # DOWN
            elif action == 2:
                new_dir = (-1, 0)  # LEFT
            elif action == 3:
                new_dir = (1, 0)   # RIGHT
            
            snake.change_direction(new_dir)
            
            # Move snake
            done = not snake.move()
            
            # Check if food was eaten
            ate_food = snake.eat_food(food)
            if ate_food:
                food = generate_food(snake.body)
                reward = 10
            else:
                reward = -0.1  # Small penalty for each step to encourage efficiency
            
            # Additional rewards/penalties
            if done:
                reward = -10
            
            # Get next state
            next_state = get_state(snake, food)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train the agent
            agent.replay(batch_size)
            
            # Early stopping if stuck
            if steps > 500:
                break
        
        # Print progress
        print(f"Episode: {e+1}/{episodes}, Score: {len(snake.body)-3}, Epsilon: {agent.epsilon:.2f}, Reward: {total_reward:.1f}")
        
        # Save model periodically
        if (e + 1) % 100 == 0:
            agent.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    agent.save_model(save_path)
    print(f"Final model saved to {save_path}")
    return agent

if __name__ == "__main__":
    # Train the DQN agent
    trained_agent = train_dqn(episodes=1000)