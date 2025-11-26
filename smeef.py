"""smeef.py

Entry/demo script for the SMEEF environment. This file provides a polished
visual demo + optional model playback. High-level structure:

- Configuration (ALGORITHM, MODEL_PATHS, visual constants)
- Particle class + visual helpers
- Utility helpers (safe_float, safe_rect_args)
- Model loader: load_model()
- Rendering: draw_grid(), draw_animated_agent(), draw_resource_section(), etc.
- Demo runner: run_demo()
- Main launcher: prints banner and calls run_demo()

Run with:

    python smeef.py

"""
import pygame
import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
import math
import random

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.smeef_env import SMEEFEnv, Action

# ------------------------
# ENHANCED CONFIG - VISUAL STORYTELLING
# ------------------------
ALGORITHM = "ppo"  
MODEL_PATHS = [
    "models/ppo/ppo_best_model",
    "models/ppo/best_model", 
    "models/ppo/ppo_model",
    "models/best_model"
]
EPISODES = 2
CELL_SIZE = 90
MARGIN = 3
AGENT_COLOR = (0, 0, 0)

# Enhanced color palette
COLORS = {
    "background": (245, 245, 250),
    "panel_bg": (255, 255, 255),
    "panel_border": (200, 200, 220),
    "text_dark": (50, 50, 70),
    "text_light": (100, 100, 130),
    "positive": (76, 175, 80),
    "warning": (255, 152, 0),
    "critical": (244, 67, 54),
    "highlight": (156, 39, 176)
}

# Service locations with enhanced visuals - COMPATIBLE WITH BOTH CASES
SERVICE_STORIES = {
    "childcare": {"emoji": "🏠", "story": "Safe childcare\n= Work freedom", "color": (173, 216, 230)},
    "education": {"emoji": "📚", "story": "Learn new skills\n= Better jobs", "color": (144, 238, 144)},
    "financial": {"emoji": "💰", "story": "Financial aid\n= Breathing room", "color": (255, 255, 150)},
    "healthcare": {"emoji": "🏥", "story": "Healthcare\n= Family wellness", "color": (255, 182, 193)},
    "community": {"emoji": "👥", "story": "Community\n= Emotional support", "color": (221, 160, 221)},
    "counseling": {"emoji": "💬", "story": "Counseling\n= Mental strength", "color": (152, 251, 152)},
    # UPPERCASE versions for compatibility
    "CHILDCARE": {"emoji": "🏠", "story": "Safe childcare\n= Work freedom", "color": (173, 216, 230)},
    "EDUCATION": {"emoji": "📚", "story": "Learn new skills\n= Better jobs", "color": (144, 238, 144)},
    "FINANCIAL": {"emoji": "💰", "story": "Financial aid\n= Breathing room", "color": (255, 255, 150)},
    "HEALTHCARE": {"emoji": "🏥", "story": "Healthcare\n= Family wellness", "color": (255, 182, 193)},
    "COMMUNITY": {"emoji": "👥", "story": "Community\n= Emotional support", "color": (221, 160, 221)},
    "COUNSELING": {"emoji": "💬", "story": "Counseling\n= Mental strength", "color": (152, 251, 152)}
}

# Particle effects for visual feedback
particles = []

class Particle:
    def __init__(self, x, y, color, effect_type="sparkle"):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = 30
        self.age = 0
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.effect_type = effect_type
        self.size = random.randint(2, 5)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        self.vy += 0.1  # gravity
        return self.age < self.lifetime
        
    def draw(self, screen):
        alpha = 255 * (1 - self.age / self.lifetime)
        if self.effect_type == "sparkle":
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        elif self.effect_type == "glow":
            radius = self.size * (1 + math.sin(self.age * 0.3))
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, int(alpha)), (radius, radius), radius)
            screen.blit(s, (int(self.x-radius), int(self.y-radius)))

# ------------------------
# UTILITY FUNCTIONS - ADDED FOR ROBUSTNESS
# ------------------------
def safe_float(value, default=0.0):
    """Safely convert value to float, handling NaN and infinity"""
    try:
        if hasattr(value, 'item'):  # Handle numpy arrays
            value = value.item()
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default

def safe_rect_args(x, y, width, height):
    """Ensure all rectangle arguments are valid numbers"""
    return (
        safe_float(x, 0),
        safe_float(y, 0),
        max(0, safe_float(width, 0)),
        max(0, safe_float(height, 0))
    )

# ------------------------
# ENHANCED LOAD MODEL
# ------------------------
def load_model():
    """Load the best available model with cinematic flair"""
    print("🚀 Initializing AI Empowerment Agent...")
    
    for model_path in MODEL_PATHS:
        full_path = model_path + ".zip"
        if os.path.exists(full_path):
            try:
                print(f"🎯 Loading trained strategy model: {full_path}")
                pygame.time.delay(500)  # Dramatic pause
                
                if ALGORITHM.lower() == "dqn":
                    model = DQN.load(model_path)
                elif ALGORITHM.lower() == "ppo":
                    model = PPO.load(model_path)
                elif ALGORITHM.lower() == "a2c":
                    model = A2C.load(model_path)
                
                print("✅ AI Strategy loaded successfully!")
                return model
            except Exception as e:
                print(f"❌ Error loading {full_path}: {e}")
                continue
    
    print("🎲 No trained strategy found. Agent will learn through exploration.")
    return None

# ------------------------
# OBSERVATION COMPATIBILITY FIX
# ------------------------
def get_model_compatible_observation(env):
    """
    Convert current environment state to model-compatible observation.
    This handles the case where the model was trained with a different observation structure.
    """
    try:
        # Try to get observation using the environment's normal method
        obs, _ = env.reset() if hasattr(env, 'get_observation') else (env._get_obs(), {})
        return obs
    except Exception as e:
        print(f"⚠️ Observation compatibility issue: {e}")
        print("🔄 Creating compatible observation structure...")
        
        # Create a simple flattened observation as fallback
        observation = np.concatenate([
            env.position,
            env.resources,
            env.needs,
            env.child_status,
            [env.current_step]
        ]).astype(np.float32)
        
        return observation

# ------------------------
# ENHANCED GRID WITH STORYTELLING - FIXED
# ------------------------
def draw_grid(screen, env, font, last_reward=0):
    screen.fill(COLORS["background"])
    
    # Animated background elements
    draw_animated_background(screen)
    
    grid_size = env.grid_size

    # Draw grid with enhanced styling
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y*(CELL_SIZE+MARGIN) + 20, x*(CELL_SIZE+MARGIN) + 20, CELL_SIZE, CELL_SIZE)
            pos = (x, y)
            
            # Enhanced cell styling with shadows
            pygame.draw.rect(screen, (230, 230, 240), rect, border_radius=8)
            pygame.draw.rect(screen, (210, 210, 220), rect, width=2, border_radius=8)
            
            label = ""
            cell_story = ""
            cell_color = (240, 240, 245)

            # Check service locations with enhanced logic - FIXED CASE SENSITIVITY
            for service_name, service_info in env.services.items():
                if pos in service_info['positions']:
                    # Use service_name directly (handles both cases)
                    service_data = SERVICE_STORIES.get(service_name)
                    if service_data:
                        cell_color = service_data["color"]
                        label = service_data["emoji"]
                        cell_story = service_data["story"]
                        # Enhanced service cell
                        pygame.draw.rect(screen, cell_color, rect, border_radius=8)
                        pygame.draw.rect(screen, (180, 180, 200), rect, width=2, border_radius=8)
                    break

            # Special locations with storytelling
            if pos == tuple(env.home_location):
                label = "🏡"
                cell_story = "Home: Rest &\nFamily time"
                pygame.draw.rect(screen, (100, 200, 100), rect, border_radius=8)
            elif pos == tuple(env.work_location):
                label = "💼"
                cell_story = "Work: Income &\nCareer growth"
                pygame.draw.rect(screen, (100, 150, 255), rect, border_radius=8)
            elif pos == tuple(env.goal_location):
                label = "⭐"
                cell_story = "Life Goal:\nStability & Success"
                # Animated goal cell
                pulse = math.sin(pygame.time.get_ticks() * 0.01) * 0.2 + 0.8
                goal_color = (255, 215, 0)
                animated_color = tuple(min(255, int(c * pulse)) for c in goal_color)
                pygame.draw.rect(screen, animated_color, rect, border_radius=10)
                pygame.draw.rect(screen, (255, 165, 0), rect, width=3, border_radius=10)

            # Draw label with enhanced styling
            if label:
                try:
                    emoji_font = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE - 30)
                    text = emoji_font.render(label, True, (50, 50, 70))
                except:
                    emoji_font = pygame.font.SysFont("Arial", CELL_SIZE - 30)
                    text = emoji_font.render(label, True, (50, 50, 70))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

            # Draw agent with animation
            if tuple(env.position) == pos:
                draw_animated_agent(screen, rect, env)

            # Show story tooltip on hover (simplified)
            mouse_pos = pygame.mouse.get_pos()
            if rect.collidepoint(mouse_pos) and cell_story:
                draw_tooltip(screen, cell_story, mouse_pos)

    # Update and draw particles
    global particles
    particles = [p for p in particles if p.update()]
    for particle in particles:
        particle.draw(screen)
    
    # Add particles for positive rewards
    if last_reward > 1.0:
        agent_rect = pygame.Rect(
            env.position[1]*(CELL_SIZE+MARGIN) + 20 + CELL_SIZE//2,
            env.position[0]*(CELL_SIZE+MARGIN) + 20 + CELL_SIZE//2,
            1, 1
        )
        for _ in range(5):
            particles.append(Particle(
                agent_rect.centerx, agent_rect.centery,
                COLORS["positive"], "sparkle"
            ))

    # Enhanced status panel
    draw_enhanced_status_panel(screen, env, grid_size)

def draw_animated_agent(screen, rect, env):
    """Draw the mother agent with personality and animation"""
    # Pulsing effect
    pulse = math.sin(pygame.time.get_ticks() * 0.005) * 0.1 + 0.9
    size = int((CELL_SIZE - 20) * pulse)
    
    # Create agent surface with transparency
    agent_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Draw agent character
    try:
        emoji_font = pygame.font.SysFont("Segoe UI Emoji", size)
        agent_text = emoji_font.render("👩", True, (50, 50, 70))
    except:
        emoji_font = pygame.font.SysFont("Arial", size)
        agent_text = emoji_font.render("M", True, (50, 50, 70))  # Fallback
    
    text_rect = agent_text.get_rect(center=(size//2, size//2))
    agent_surface.blit(agent_text, text_rect)
    
    # Position on screen
    screen_rect = agent_surface.get_rect(center=rect.center)
    screen.blit(agent_surface, screen_rect)
    
    # Draw agent trail
    draw_agent_trail(screen, env)

def draw_agent_trail(screen, env):
    """Show where the agent has been"""
    trail_positions = getattr(env, 'trail_positions', [])
    trail_positions.append(tuple(env.position))
    
    # Keep only recent positions
    if len(trail_positions) > 5:
        trail_positions.pop(0)
    env.trail_positions = trail_positions
    
    # Draw trail
    for i, pos in enumerate(trail_positions[:-1]):  # Exclude current position
        alpha = int(100 * (i / len(trail_positions)))
        trail_rect = pygame.Rect(
            pos[1]*(CELL_SIZE+MARGIN) + 20 + CELL_SIZE//4,
            pos[0]*(CELL_SIZE+MARGIN) + 20 + CELL_SIZE//4,
            CELL_SIZE//2, CELL_SIZE//2
        )
        trail_surface = pygame.Surface((CELL_SIZE//2, CELL_SIZE//2), pygame.SRCALPHA)
        pygame.draw.circle(trail_surface, (100, 100, 200, alpha), 
                         (CELL_SIZE//4, CELL_SIZE//4), CELL_SIZE//6)
        screen.blit(trail_surface, trail_rect)

def draw_animated_background(screen):
    """Add subtle animated background elements"""
    time = pygame.time.get_ticks() * 0.001
    
    # Floating particles in background
    for i in range(5):
        x = (math.sin(time + i) * 0.5 + 0.5) * screen.get_width()
        y = (math.cos(time * 0.7 + i) * 0.5 + 0.5) * screen.get_height() * 0.3
        size = 2 + math.sin(time + i) * 1
        pygame.draw.circle(screen, (200, 200, 220, 100), (int(x), int(y)), int(size))

def draw_tooltip(screen, text, pos):
    """Draw a tooltip with the cell's story"""
    font = pygame.font.SysFont("Arial", 12)
    lines = text.split('\n')
    
    # Calculate tooltip size
    max_width = max(font.size(line)[0] for line in lines)
    height = len(lines) * 18
    
    tooltip_rect = pygame.Rect(pos[0] + 10, pos[1] + 10, max_width + 20, height + 10)
    
    # Draw tooltip
    pygame.draw.rect(screen, (255, 255, 240), tooltip_rect, border_radius=4)
    pygame.draw.rect(screen, (200, 200, 180), tooltip_rect, width=1, border_radius=4)
    
    # Draw text
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, COLORS["text_dark"])
        screen.blit(text_surface, (tooltip_rect.x + 10, tooltip_rect.y + 5 + i * 18))

def draw_enhanced_status_panel(screen, env, grid_size):
    """Create a cinematic status panel that tells a story"""
    panel_x = grid_size * (CELL_SIZE + MARGIN) + 40
    panel_width = 350
    panel_height = screen.get_height() - 40
    
    # Main panel with styling
    panel_rect = pygame.Rect(panel_x, 20, panel_width, panel_height)
    pygame.draw.rect(screen, COLORS["panel_bg"], panel_rect, border_radius=12)
    pygame.draw.rect(screen, COLORS["panel_border"], panel_rect, width=2, border_radius=12)
    
    title_font = pygame.font.SysFont("Arial", 24, bold=True)
    section_font = pygame.font.SysFont("Arial", 18, bold=True)
    value_font = pygame.font.SysFont("Arial", 16)
    small_font = pygame.font.SysFont("Arial", 12)
    
    y_offset = 40
    
    # Main title with algorithm badge
    title = title_font.render("Mother's Journey Dashboard", True, COLORS["text_dark"])
    screen.blit(title, (panel_x + 20, y_offset))
    
    algo_badge = small_font.render(f"STRATEGY: {ALGORITHM.upper()}", True, COLORS["highlight"])
    screen.blit(algo_badge, (panel_x + panel_width - 120, y_offset + 5))
    y_offset += 50
    
    # Resources section with icons
    draw_resource_section(screen, env, panel_x, y_offset, section_font, value_font)
    y_offset += 150
    
    # Needs section with urgency indicators
    draw_needs_section(screen, env, panel_x, y_offset, section_font, value_font)
    y_offset += 180
    
    # Child status with emotional indicators
    draw_child_section(screen, env, panel_x, y_offset, section_font, value_font)
    y_offset += 100
    
    # Progress and narrative
    draw_progress_section(screen, env, panel_x, y_offset, value_font, small_font)

def draw_resource_section(screen, env, x, y, section_font, value_font):
    """Draw resources with visual indicators - COMPLETELY FIXED"""
    title_text = section_font.render("Resources & Strengths", True, COLORS["text_dark"])
    screen.blit(title_text, (x + 20, y))
    y += 30
    
    resources = [
        ("💰", "Money", env.resources[0], 100, "Financial security"),
        ("⚡", "Energy", env.resources[1], 10, "Daily capacity"), 
        ("🎯", "Skills", env.resources[2], 100, "Earning potential"),
        ("🤝", "Support", env.resources[3], 100, "Social network")
    ]
    
    for emoji, name, value, max_val, description in resources:
        # SAFELY get and validate values
        safe_value = safe_float(value, 0)
        safe_max_val = safe_float(max_val, 100)
        
        # Resource bar dimensions
        bar_width = 250
        bar_height = 6
        bar_x = safe_float(x + 20, x + 20)
        bar_y = safe_float(y + 20, y + 20)
        
        # Background bar - always draw this
        bg_rect_args = safe_rect_args(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(screen, (220, 220, 220), bg_rect_args, border_radius=3)
        
        # Calculate fill width SAFELY
        if safe_max_val > 0 and safe_value >= 0:
            fill_ratio = min(safe_value / safe_max_val, 1.0)
            fill_width = fill_ratio * bar_width
        else:
            fill_width = 0
            
        # Ensure fill_width is valid
        safe_fill_width = max(0, min(fill_width, bar_width))
        
        # Choose color based on value
        if safe_value > safe_max_val * 0.3:
            color = COLORS["positive"]
        elif safe_value > safe_max_val * 0.1:
            color = COLORS["warning"]
        else:
            color = COLORS["critical"]
        
        # Only draw fill bar if we have a positive width
        if safe_fill_width > 0:
            fill_rect_args = safe_rect_args(bar_x, bar_y, safe_fill_width, bar_height)
            pygame.draw.rect(screen, color, fill_rect_args, border_radius=3)
        
        # Text labels
        try:
            emoji_font = pygame.font.SysFont("Segoe UI Emoji", 20)
            emoji_text = emoji_font.render(emoji, True, COLORS["text_dark"])
        except:
            emoji_font = value_font
            emoji_text = emoji_font.render(emoji, True, COLORS["text_dark"])
        screen.blit(emoji_text, (x + 20, y))
        
        info_text = value_font.render(f"{name}: {safe_value:.1f}", True, COLORS["text_dark"])
        screen.blit(info_text, (x + 50, y))
        
        y += 30

def draw_needs_section(screen, env, x, y, section_font, value_font):
    """Draw needs with urgency visualization - COMPLETELY FIXED"""
    title_text = section_font.render("Pressures & Needs", True, COLORS["text_dark"])
    screen.blit(title_text, (x + 20, y))
    y += 30
    
    needs = [
        ("👶", "Childcare", env.needs[0], "Child supervision"),
        ("💸", "Financial", env.needs[1], "Bills & expenses"),
        ("💖", "Emotional", env.needs[2], "Mental wellness"), 
        ("💼", "Career", env.needs[3], "Job advancement")
    ]
    
    for emoji, name, value, description in needs:
        # SAFELY get and validate value
        safe_value = safe_float(value, 50)
        safe_value = max(0, min(safe_value, 100))  # Clamp between 0-100
        
        # Urgency indicator (circle)
        indicator_size = 20
        indicator_x = safe_float(x + 20, x + 20)
        indicator_y = safe_float(y + 10, y + 10)
        
        # Color based on urgency
        if safe_value > 70:
            color = COLORS["critical"]
            pulse = math.sin(pygame.time.get_ticks() * 0.01) * 0.3 + 0.7
            size = int(indicator_size * pulse)
        elif safe_value > 40:
            color = COLORS["warning"]
            size = indicator_size
        else:
            color = COLORS["positive"] 
            size = indicator_size
            
        # Draw indicator circle
        pygame.draw.circle(screen, color, (int(indicator_x), int(indicator_y)), size)
        pygame.draw.circle(screen, COLORS["text_dark"], (int(indicator_x), int(indicator_y)), size, 1)
        
        # Text labels
        try:
            emoji_font = pygame.font.SysFont("Segoe UI Emoji", 20)
            emoji_text = emoji_font.render(emoji, True, COLORS["text_dark"])
        except:
            emoji_font = value_font
            emoji_text = emoji_font.render(emoji, True, COLORS["text_dark"])
        screen.blit(emoji_text, (x + 50, y))
        
        need_text = value_font.render(f"{name}: {safe_value:.1f}", True, COLORS["text_dark"])
        screen.blit(need_text, (x + 80, y))
        
        y += 35

def draw_child_section(screen, env, x, y, section_font, value_font):
    """Draw child status with emotional indicators - COMPLETELY FIXED"""
    title_text = section_font.render("Child's Well-being", True, COLORS["text_dark"])
    screen.blit(title_text, (x + 20, y))
    y += 30
    
    # SAFELY get child status values
    health = safe_float(env.child_status[0], 50)
    happiness = safe_float(env.child_status[1], 50)
    
    # Clamp values
    health = max(0, min(health, 100))
    happiness = max(0, min(happiness, 100))
    
    # Child face based on happiness
    if happiness > 75:
        face = "😊"
    elif happiness > 50:
        face = "😐" 
    elif happiness > 25:
        face = "😟"
    else:
        face = "😢"
    
    try:
        face_font = pygame.font.SysFont("Segoe UI Emoji", 40)
        face_text = face_font.render(face, True, COLORS["text_dark"])
    except:
        face_font = pygame.font.SysFont("Arial", 24)
        face_text = face_font.render(face, True, COLORS["text_dark"])
    screen.blit(face_text, (x + 150, y))
    
    stats = [
        f"Health: {health:.1f}",
        f"Happiness: {happiness:.1f}"
    ]
    
    for stat in stats:
        text = value_font.render(stat, True, COLORS["text_dark"])
        screen.blit(text, (x + 20, y))
        y += 25

def draw_progress_section(screen, env, x, y, value_font, small_font):
    """Draw progress and narrative - COMPLETELY FIXED"""
    # SAFELY calculate progress
    safe_current_step = safe_float(env.current_step, 0)
    safe_max_steps = safe_float(env.max_steps, 50)
    
    if safe_max_steps > 0:
        progress = safe_current_step / safe_max_steps
    else:
        progress = 0
    progress = max(0, min(progress, 1))  # Clamp between 0 and 1
    
    bar_width = 250
    bar_height = 12
    bar_x = safe_float(x + 20, x + 20)
    bar_y = safe_float(y, y)
    
    # Background progress bar
    bg_rect_args = safe_rect_args(bar_x, bar_y, bar_width, bar_height)
    pygame.draw.rect(screen, (220, 220, 220), bg_rect_args, border_radius=6)
    
    # Progress fill bar
    if progress > 0:
        progress_width = progress * bar_width
        fill_rect_args = safe_rect_args(bar_x, bar_y, progress_width, bar_height)
        pygame.draw.rect(screen, COLORS["highlight"], fill_rect_args, border_radius=6)
    
    progress_text = value_font.render(f"Day {int(safe_current_step)} of {int(safe_max_steps)}", True, COLORS["text_dark"])
    screen.blit(progress_text, (bar_x, bar_y + 15))
    
    # Narrative based on performance
    y += 40
    narrative = generate_narrative(env)
    narrative_font = pygame.font.SysFont("Arial", 12)
    
    # Word wrap for narrative
    words = narrative.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if narrative_font.size(test_line)[0] < 310:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    for i, line in enumerate(lines):
        text = narrative_font.render(line, True, COLORS["text_light"])
        screen.blit(text, (x + 20, y + i * 16))

def generate_narrative(env):
    """Generate a dynamic narrative based on the current state"""
    # Use safe values
    money = safe_float(env.resources[0], 0)
    happiness = safe_float(env.child_status[1], 50)
    current_step = safe_float(env.current_step, 0)
    max_steps = safe_float(env.max_steps, 50)
    
    if current_step < max_steps * 0.3:
        phase = "Early Journey"
    elif current_step < max_steps * 0.7:
        phase = "Building Momentum" 
    else:
        phase = "Final Stretch"
        
    if money < 20:
        financial = "Financial pressure is high"
    elif money < 50:
        financial = "Making financial progress"
    else:
        financial = "Financially stable"
        
    if happiness < 40:
        child = "Child needs more attention"
    elif happiness < 70:
        child = "Child is doing okay"
    else:
        child = "Child is thriving"
    
    return f"{phase}: {financial}. {child}. Every choice matters."

# ------------------------
# ENHANCED DEMO EXECUTION - FIXED DISPLAY LOOP
# ------------------------
def run_demo():
    pygame.init()
    env = SMEEFEnv(grid_size=6, max_steps=50)
    grid_size = env.grid_size
    screen_width = grid_size * (CELL_SIZE + MARGIN) + 420
    screen_height = grid_size * (CELL_SIZE + MARGIN) + 40
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("🌟 SMEEF: The Single Mother Empowerment Journey")
    
    # Set window icon
    try:
        icon = pygame.Surface((32, 32))
        icon.fill((100, 150, 255))
        pygame.display.set_icon(icon)
    except:
        pass
    
    clock = pygame.time.Clock()

    # Load model
    model = load_model()
    using_trained_model = model is not None

    font = pygame.font.SysFont("Arial", 16)

    print("\n" + "✨" * 60)
    print("           SMEEF CINEMATIC DEMONSTRATION")
    print("   Witness the AI-Driven Empowerment Journey")
    print("✨" * 60)

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        last_reward = 0

        print(f"\n🎬 EPISODE {ep + 1}: {'AI OPTIMIZED STRATEGY' if using_trained_model else 'EXPLORATORY LEARNING'}")
        print("📖 Story: A mother's journey to balance family, work, and personal growth")
        print("─" * 70)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # Get action from model or random with compatibility handling
            if model:
                try:
                    # Get model-compatible observation
                    model_obs = get_model_compatible_observation(env)
                    action, _ = model.predict(model_obs, deterministic=True)
                    action_name = Action(action).name
                except Exception as e:
                    print(f"⚠️ Model prediction failed: {e}")
                    print("🔄 Falling back to random actions...")
                    action = env.action_space.sample()
                    action_name = Action(action).name
            else:
                action = env.action_space.sample()
                action_name = Action(action).name

            # Take environment step
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            total_reward += reward
            last_reward = reward
            done = terminated or truncated

            # 🎯 CRITICAL FIX: CONTINUOUSLY UPDATE THE DISPLAY
            draw_grid(screen, env, font, last_reward)
            pygame.display.flip()  # This makes the display actually update!
            
            # Print action details
            print(f"Step {step_count:2d}: {action_name:20} | Reward: {reward:7.2f} | Position: {env.position}")
            
            # Control speed for better visualization
            clock.tick(2)  # 2 FPS - slow enough to see the movement

        # Episode results
        print(f"\n🏁 EPISODE {ep + 1} COMPLETE!")
        print(f"   📊 Total Empowerment Score: {total_reward:7.2f}")
        print(f"   🕒 Days Survived: {step_count}")
        print(f"   💰 Final Resources: ${env.resources[0]:.1f}")
        print(f"   ❤️  Child's Happiness: {env.child_status[1]:.1f}")
        
        # Performance assessment
        if total_reward > 50:
            print("   🌟 OUTSTANDING: Mother is thriving!")
        elif total_reward > 0:
            print("   ✅ SUCCESS: Positive progress made!")
        else:
            print("   💔 CHALLENGING: The struggle continues...")

        # Brief pause between episodes
        if ep < EPISODES - 1:
            print("\n🔄 Preparing next episode...")
            pygame.time.delay(2000)  # 2 second pause

    print("\n" + "🎉" * 60)
    print("           DEMONSTRATION COMPLETE!")
    print("   This AI doesn't just play games - it understands life.")
    print("🎉" * 60)
    
    # Keep window open for a moment
    pygame.time.delay(3000)
    pygame.quit()

# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    print("\n" + "🌍" * 70)
    print("           WELCOME TO SMEEF: SINGLE MOTHER EMPOWERMENT FRAMEWORK")
    print("   Where Artificial Intelligence Meets Human Resilience")
    print("🌍" * 70)
    
    print("\n📖 THE STORY:")
    print("   You are about to witness an AI agent learning to navigate the complex")
    print("   challenges faced by single mothers - balancing work, childcare,")
    print("   personal growth, and emotional well-being.")
    
    print("\n🎯 THE MISSION:")
    print("   • Balance financial stability with family care")
    print("   • Grow skills while maintaining emotional health") 
    print("   • Build support networks and access resources")
    print("   • Ultimately, achieve sustainable empowerment")
    
    print("\n🏗️  THE TECHNOLOGY:")
    print("   • Reinforcement Learning for life strategy optimization")
    print("   • Multi-objective balancing of competing priorities")
    print("   • Ethical AI that understands human constraints")
    
    print("\n" + "─" * 70)
    print("Press ESC to exit at any time | Watch the AI learn and adapt!")
    print("─" * 70)
    
    input("\nPress Enter to begin this inspiring journey...")
    
    run_demo()