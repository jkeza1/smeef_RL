# smeef_gui_color_agent.py
import pygame
import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from environment.smeef_env import SMEEFEnv

# ------------------------
# CONFIG
# ------------------------
ALGORITHM = "dqn"
MODEL_PATH = "models/dqn/best_model"
EPISODES = 3
CELL_SIZE = 80
MARGIN = 2
AGENT_COLOR = (0, 0, 0)  # black for emoji

# ------------------------
# LOAD MODEL
# ------------------------
def load_model(algo, path):
    if not os.path.exists(path + ".zip"):
        print(f"WARNING: Model file {path}.zip not found. Running random agent.\n")
        return None
    if algo.lower() == "dqn":
        return DQN.load(path)
    elif algo.lower() == "ppo":
        return PPO.load(path)
    elif algo.lower() == "a2c":
        return A2C.load(path)
    else:
        raise ValueError("Unsupported algorithm!")
    
def heuristic_move(agent_pos, env):
    targets = env.skill_centers + env.income_ops + env.community_support + env.challenges + [env.goal]
    ax, ay = agent_pos.astype(int)
    distances = [abs(ax-tx)+abs(ay-ty) for tx,ty in targets]
    tx, ty = targets[np.argmin(distances)]
    if ax < tx:
        return 1  # DOWN
    elif ax > tx:
        return 0  # UP
    elif ay < ty:
        return 3  # RIGHT
    elif ay > ty:
        return 2  # LEFT
    else:
        return np.random.choice([0,1,2,3])


# ------------------------
# DRAW GRID
# ------------------------
def draw_grid(screen, env, font):
    screen.fill((255, 255, 255))  # background

    # Colors for different objects
    color_map = {
        "skill": (255, 255, 0),      # yellow 🟨
        "income": (0, 0, 255),       # blue 🟦
        "community": (128, 0, 128),  # purple 🟪
        "challenge": (255, 0, 0),    # red 🟥
        "goal": (0, 255, 0),         # green 🟩
        "empty": (220, 220, 220)     # light gray ⬜
    }

    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(y*(CELL_SIZE+MARGIN), x*(CELL_SIZE+MARGIN), CELL_SIZE, CELL_SIZE)
            pos = (x, y)
            color = color_map["empty"]

            if pos in env.skill_centers:
                color = color_map["skill"]
            elif pos in env.income_ops:
                color = color_map["income"]
            elif pos in env.community_support:
                color = color_map["community"]
            elif pos in env.challenges:
                color = color_map["challenge"]
            elif pos == env.goal:
                color = color_map["goal"]

            pygame.draw.rect(screen, color, rect)

            # Draw agent emoji on top
            if tuple(env.agent_pos.astype(int)) == pos:
                text = font.render("🤖", True, AGENT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

# ------------------------
# RUN DEMO
# ------------------------
def run_demo(model=None, episodes=EPISODES):
    pygame.init()
    env = SMEEFEnv(grid_size=6)
    screen_size = env.grid_size * CELL_SIZE + (env.grid_size - 1) * MARGIN
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("SMEEF GUI Demo (Color + Agent)")
    clock = pygame.time.Clock()

    # Font for agent emoji
    font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 10)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        env.total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Action
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            env.total_reward += reward
            done = terminated or truncated

            draw_grid(screen, env, font)
            pygame.display.flip()
            clock.tick(3)  # 3 steps/sec

            print(f"Step: {step_count} | Action: {action} | Reward: {reward}")

        print(f"\nEpisode {ep + 1} finished with total reward: {env.total_reward}\n")

    pygame.quit()
    print("Demo finished.")

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    print("\n==============================")
    print("   SMEEF GUI DEMO (Color + Agent)")
    print("==============================\n")
    model = load_model(ALGORITHM, MODEL_PATH)
    run_demo(model=model)
