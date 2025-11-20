import pygame
import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from environment.smeef_env import SMEEFEnv, Action

# ------------------------
# CONFIG
# ------------------------
ALGORITHM = "dqn"
MODEL_PATH = "models/dqn/best_model.zip"  # Your trained model path
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

# ------------------------
# DRAW GRID - UPDATED FOR COMPLEX ENVIRONMENT
# ------------------------
def draw_grid(screen, env, font):
    screen.fill((255, 255, 255))  # background

    # Colors for different services and locations
    color_map = {
        "childcare": (173, 216, 230),    # Light blue
        "education": (144, 238, 144),    # Light green
        "financial": (255, 255, 150),    # Light yellow
        "healthcare": (255, 182, 193),   # Light pink
        "community": (221, 160, 221),    # Light purple
        "counseling": (152, 251, 152),   # Pale green
        "home": (100, 200, 100),         # Green
        "work": (100, 150, 255),         # Blue
        "goal": (255, 215, 0),           # Gold
        "empty": (220, 220, 220)         # Light gray
    }

    # Emojis for different locations
    emoji_map = {
        "childcare": "🏠",
        "education": "📚",
        "financial": "💰",
        "healthcare": "🏥",
        "community": "👥",
        "counseling": "💬",
        "home": "🏡",
        "work": "💼",
        "goal": "⭐"
    }

    grid_size = env.grid_size

    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y*(CELL_SIZE+MARGIN), x*(CELL_SIZE+MARGIN), CELL_SIZE, CELL_SIZE)
            pos = (x, y)
            color = color_map["empty"]
            emoji = ""

            # Check service locations
            for service_name, service_info in env.services.items():
                if pos in service_info['positions']:
                    color = service_info['color']
                    emoji = emoji_map.get(service_name.lower(), "❓")
                    break

            # Check special locations
            if pos == env.home_location:
                color = color_map["home"]
                emoji = emoji_map["home"]
            elif pos == env.work_location:
                color = color_map["work"]
                emoji = emoji_map["work"]
            elif pos == env.goal_location:
                color = color_map["goal"]
                emoji = emoji_map["goal"]

            pygame.draw.rect(screen, color, rect)

            # Draw emoji if this is a special location
            if emoji:
                small_font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 40)
                text = small_font.render(emoji, True, AGENT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

            # Draw agent emoji on top
            if tuple(env.position) == pos:
                agent_font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 20)
                agent_text = agent_font.render("👩", True, AGENT_COLOR)
                agent_rect = agent_text.get_rect(center=rect.center)
                screen.blit(agent_text, agent_rect)

    # Draw status panel
    draw_status_panel(screen, env, grid_size)

def draw_status_panel(screen, env, grid_size):
    panel_x = grid_size * (CELL_SIZE + MARGIN) + 10
    panel_width = 300
    small_font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 20, bold=True)
    
    y_offset = 20
    
    # Title
    title = title_font.render("Single Mother Status", True, (0, 0, 0))
    screen.blit(title, (panel_x, y_offset))
    y_offset += 40
    
    # Resources
    resources_title = title_font.render("Resources:", True, (0, 0, 0))
    screen.blit(resources_title, (panel_x, y_offset))
    y_offset += 30
    
    resources = [
        f"💰 Money: {env.resources[0]:.1f}",
        f"⚡ Energy: {env.resources[1]:.1f}",
        f"🎯 Skills: {env.resources[2]:.1f}",
        f"🤝 Support: {env.resources[3]:.1f}"
    ]
    
    for resource in resources:
        text = small_font.render(resource, True, (0, 0, 0))
        screen.blit(text, (panel_x, y_offset))
        y_offset += 25
    
    y_offset += 10
    
    # Needs
    needs_title = title_font.render("Needs:", True, (0, 0, 0))
    screen.blit(needs_title, (panel_x, y_offset))
    y_offset += 30
    
    needs = [
        f"👶 Childcare: {env.needs[0]:.1f}",
        f"💸 Financial: {env.needs[1]:.1f}",
        f"💗 Emotional: {env.needs[2]:.1f}",
        f"📈 Career: {env.needs[3]:.1f}"
    ]
    
    for need in needs:
        text = small_font.render(need, True, (0, 0, 0))
        screen.blit(text, (panel_x, y_offset))
        y_offset += 25
    
    y_offset += 10
    
    # Child Status
    child_title = title_font.render("Child Status:", True, (0, 0, 0))
    screen.blit(child_title, (panel_x, y_offset))
    y_offset += 30
    
    child_stats = [
        f"🏥 Health: {env.child_status[0]:.1f}",
        f"😊 Happiness: {env.child_status[1]:.1f}"
    ]
    
    for stat in child_stats:
        text = small_font.render(stat, True, (0, 0, 0))
        screen.blit(text, (panel_x, y_offset))
        y_offset += 25
    
    y_offset += 10
    
    # Step counter
    step_text = small_font.render(f"Step: {env.current_step}/{env.max_steps}", True, (0, 0, 0))
    screen.blit(step_text, (panel_x, y_offset))

# ------------------------
# RUN DEMO
# ------------------------
def run_demo(model=None, episodes=EPISODES):
    pygame.init()
    env = SMEEFEnv(grid_size=6, max_steps=100)
    grid_size = env.grid_size
    screen_width = grid_size * (CELL_SIZE + MARGIN) + 320  # Extra space for status panel
    screen_height = grid_size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SMEEF Demo - Single Mother Empowerment")
    clock = pygame.time.Clock()

    # Font for agent emoji
    font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 10)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"\n=== Episode {ep + 1} ===")
        print(f"Starting at home 🏡")
        print(f"Initial resources - Money: ${env.resources[0]:.1f}, Energy: {env.resources[1]:.1f}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Action
            if model:
                action, _ = model.predict(obs, deterministic=True)
                action_name = Action(action).name
            else:
                action = env.action_space.sample()
                action_name = "RANDOM"

            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            total_reward += reward
            done = terminated or truncated

            draw_grid(screen, env, font)
            pygame.display.flip()
            clock.tick(2)  # 2 steps/sec for better visualization

            if step_count % 5 == 0:  # Print every 5 steps to avoid spam
                print(f"Step {step_count}: {action_name} | Reward: {reward:.2f} | Pos: {env.position}")

        print(f"Episode {ep + 1} finished!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final position: {env.position}")
        print(f"Final Money: ${env.resources[0]:.1f}, Energy: {env.resources[1]:.1f}")

    pygame.quit()
    print("\nDemo finished.")

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   SMEEF DEMO - Single Mother Empowerment")
    print("="*50)
    print("Legend:")
    print("👩 - Single Mother Agent")
    print("🏡 - Home | 💼 - Work | ⭐ - Goal")
    print("🏠 - Childcare | 📚 - Education | 💰 - Financial Aid")
    print("🏥 - Healthcare | 👥 - Community | 💬 - Counseling")
    print("\n" + "="*50)
    
    model = load_model(ALGORITHM, MODEL_PATH)
    run_demo(model=model)