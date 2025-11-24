import pygame
import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.smeef_env import SMEEFEnv, Action

# ------------------------
# CONFIG - UPDATED FOR YOUR SETUP
# ------------------------
ALGORITHM = "ppo"  # Your best algorithm
MODEL_PATHS = [
    "models/ppo/ppo_best_model",
    "models/ppo/best_model", 
    "models/ppo/ppo_model",
    "models/best_model"
]
EPISODES = 2
CELL_SIZE = 80
MARGIN = 2
AGENT_COLOR = (0, 0, 0)

# ------------------------
# LOAD MODEL - WITH BETTER ERROR HANDLING
# ------------------------
def load_model():
    """Load the best available model with fallback to random"""
    for model_path in MODEL_PATHS:
        full_path = model_path + ".zip"
        if os.path.exists(full_path):
            try:
                print(f"🔄 Loading model from: {full_path}")
                if ALGORITHM.lower() == "dqn":
                    return DQN.load(model_path)
                elif ALGORITHM.lower() == "ppo":
                    return PPO.load(model_path)
                elif ALGORITHM.lower() == "a2c":
                    return A2C.load(model_path)
            except Exception as e:
                print(f"❌ Error loading {full_path}: {e}")
                continue
    
    print("⚠️  No trained model found. Running with random actions.")
    return None

# ------------------------
# DRAW GRID - WITH FALLBACK FOR EMOJIS
# ------------------------
def draw_grid(screen, env, font):
    screen.fill((255, 255, 255))

    # Colors for different services
    color_map = {
        "childcare": (173, 216, 230),
        "education": (144, 238, 144), 
        "financial": (255, 255, 150),
        "healthcare": (255, 182, 193),
        "community": (221, 160, 221),
        "counseling": (152, 251, 152),
        "home": (100, 200, 100),
        "work": (100, 150, 255),
        "goal": (255, 215, 0),
        "empty": (220, 220, 220)
    }

    # Try to use emojis, fallback to text
    try:
        emoji_map = {
            "childcare": "🏠", "education": "📚", "financial": "💰",
            "healthcare": "🏥", "community": "👥", "counseling": "💬", 
            "home": "🏡", "work": "💼", "goal": "⭐"
        }
        use_emojis = True
    except:
        emoji_map = {
            "childcare": "CC", "education": "ED", "financial": "FI",
            "healthcare": "HC", "community": "CO", "counseling": "CN",
            "home": "HM", "work": "WK", "goal": "GL"
        }
        use_emojis = False
        print("⚠️  Emojis not available, using text labels")

    grid_size = env.grid_size

    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y*(CELL_SIZE+MARGIN), x*(CELL_SIZE+MARGIN), CELL_SIZE, CELL_SIZE)
            pos = (x, y)
            color = color_map["empty"]
            label = ""

            # Check service locations
            for service_name, service_info in env.services.items():
                if pos in service_info['positions']:
                    color = service_info['color']
                    label = emoji_map.get(service_name.lower(), "?")
                    break

            # Check special locations
            if pos == tuple(env.home_location):
                color = color_map["home"]
                label = emoji_map["home"]
            elif pos == tuple(env.work_location):
                color = color_map["work"] 
                label = emoji_map["work"]
            elif pos == tuple(env.goal_location):
                color = color_map["goal"]
                label = emoji_map["goal"]

            pygame.draw.rect(screen, color, rect)

            # Draw label
            if label:
                if use_emojis:
                    small_font = pygame.font.SysFont("Arial", CELL_SIZE - 40)  # Fallback font
                    try:
                        # Try to use emoji font
                        emoji_font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 40)
                        text = emoji_font.render(label, True, AGENT_COLOR)
                    except:
                        text = small_font.render(label, True, AGENT_COLOR)
                else:
                    small_font = pygame.font.SysFont("Arial", 20)
                    text = small_font.render(label, True, AGENT_COLOR)
                
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

            # Draw agent
            if tuple(env.position) == pos:
                agent_label = "👩" if use_emojis else "AM"  # Agent Mother
                if use_emojis:
                    try:
                        agent_font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 20)
                        agent_text = agent_font.render(agent_label, True, AGENT_COLOR)
                    except:
                        agent_font = pygame.font.SysFont("Arial", CELL_SIZE - 20)
                        agent_text = agent_font.render(agent_label, True, AGENT_COLOR)
                else:
                    agent_font = pygame.font.SysFont("Arial", 24)
                    agent_text = agent_font.render(agent_label, True, AGENT_COLOR)
                
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
        f"Money: ${env.resources[0]:.1f}",
        f"Energy: {env.resources[1]:.1f}",
        f"Skills: {env.resources[2]:.1f}",
        f"Support: {env.resources[3]:.1f}"
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
        f"Childcare: {env.needs[0]:.1f}",
        f"Financial: {env.needs[1]:.1f}",
        f"Emotional: {env.needs[2]:.1f}",
        f"Career: {env.needs[3]:.1f}"
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
        f"Health: {env.child_status[0]:.1f}",
        f"Happiness: {env.child_status[1]:.1f}"
    ]
    
    for stat in child_stats:
        text = small_font.render(stat, True, (0, 0, 0))
        screen.blit(text, (panel_x, y_offset))
        y_offset += 25
    
    y_offset += 10
    
    # Step counter
    step_text = small_font.render(f"Step: {env.current_step}/{env.max_steps}", True, (0, 0, 0))
    screen.blit(step_text, (panel_x, y_offset))
    
    # Algorithm info
    algo_text = small_font.render(f"Algorithm: {ALGORITHM.upper()}", True, (0, 0, 0))
    screen.blit(algo_text, (panel_x, y_offset + 30))

# ------------------------
# RUN DEMO
# ------------------------
def run_demo():
    pygame.init()
    env = SMEEFEnv(grid_size=6, max_steps=50)  # Shorter for demo
    grid_size = env.grid_size
    screen_width = grid_size * (CELL_SIZE + MARGIN) + 320
    screen_height = grid_size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SMEEF - Single Mother Empowerment Framework")
    clock = pygame.time.Clock()

    # Load model
    model = load_model()
    using_trained_model = model is not None

    font = pygame.font.SysFont("Arial", 16)  # Fallback font

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"\n" + "="*50)
        print(f"EPISODE {ep + 1} - {'TRAINED AGENT' if using_trained_model else 'RANDOM ACTIONS'}")
        print("="*50)
        print(f"Starting at home | Money: ${env.resources[0]:.1f} | Energy: {env.resources[1]:.1f}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get action
            if model:
                action, _ = model.predict(obs, deterministic=True)
                action_name = Action(action).name
            else:
                action = env.action_space.sample()
                action_name = f"RANDOM({Action(action).name})"

            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            total_reward += reward
            done = terminated or truncated

            # Update display
            draw_grid(screen, env, font)
            pygame.display.flip()
            clock.tick(2)  # 2 FPS for better visualization

            # Print progress
            if step_count % 5 == 0 or reward != 0:
                print(f"Step {step_count}: {action_name:15} | Reward: {reward:6.2f} | Pos: {env.position}")

        # Episode results
        print(f"\n📊 EPISODE {ep + 1} COMPLETE:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps: {step_count}")
        print(f"   Final Position: {env.position}")
        print(f"   Resources - Money: ${env.resources[0]:.1f}, Energy: {env.resources[1]:.1f}")
        print(f"   Child - Health: {env.child_status[0]:.1f}, Happiness: {env.child_status[1]:.1f}")

    pygame.quit()
    print("\n" + "="*50)
    print("🎬 DEMO FINISHED")
    print("="*50)

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("        SMEEF ENHANCED DEMONSTRATION")
    print("   Single Mother Empowerment Framework")
    print("="*60)
    print("🎯 GOAL: Balance childcare, work, and personal development")
    print("👩 AGENT: Single Mother navigating complex environment")
    print("🏆 BEST ALGORITHM: PPO (Proximal Policy Optimization)")
    print("\n📍 LEGEND:")
    print("   👩 - Single Mother Agent")
    print("   🏡 - Home | 💼 - Work | ⭐ - Goal") 
    print("   🏠 - Childcare | 📚 - Education | 💰 - Financial Aid")
    print("   🏥 - Healthcare | 👥 - Community | 💬 - Counseling")
    print("="*60)
    
    run_demo()