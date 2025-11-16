import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SMEEFEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=6):
        super().__init__()

        self.grid_size = grid_size
        
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation: agent position (x,y) as float32
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Agent starts at top-left
        self.agent_pos = np.array([0, 0], dtype=np.float32)

        # Object placements
        self.skill_centers = [(1, 3), (2, 1)]         # 🟨
        self.income_ops = [(3, 4)]                    # 🟦
        self.community_support = [(4, 2), (1, 4)]     # 🟪
        self.challenges = [(2, 3), (4, 1), (3, 2)]    # 🟥

        # Goal
        self.goal = (5, 5)                             # 🟩

        self.total_reward = 0
        return self.agent_pos, {}  # Gymnasium-style

    def step(self, action):
        x, y = self.agent_pos

        # Movement
        if action == 0 and x > 0:            # UP
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # DOWN
            x += 1
        elif action == 2 and y > 0:          # LEFT
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # RIGHT
            y += 1

        self.agent_pos = np.array([x, y], dtype=np.float32)
        reward = 0
        pos_tuple = (x, y)

        # REWARDS
        if pos_tuple in self.skill_centers:
            reward += 5    # 🟨 gain skills
        if pos_tuple in self.income_ops:
            reward += 10   # 🟦 income
        if pos_tuple in self.community_support:
            reward += 8    # 🟪 community help
        if pos_tuple in self.challenges:
            reward -= 5    # 🟥 challenge

        # GOAL REACHED
        terminated = pos_tuple == self.goal
        if terminated:
            reward += 50   # 🟩 empowerment zone

        truncated = False
        self.total_reward += reward

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Only human render mode is supported for SMEEFEnv.")

        # Create grid
        grid = [["⬜" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place objects
        for x, y in self.skill_centers:
            grid[x][y] = "🟨"
        for x, y in self.income_ops:
            grid[x][y] = "🟦"
        for x, y in self.community_support:
            grid[x][y] = "🟪"
        for x, y in self.challenges:
            grid[x][y] = "🟥"

        goal_x, goal_y = self.goal
        grid[goal_x][goal_y] = "🟩"

        # Place agent
        ax, ay = self.agent_pos
        ax, ay = int(ax), int(ay)
        grid[ax][ay] = "🤖"

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("-" * 20)
