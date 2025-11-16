import pygame
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)   # Goal
BLUE = (0, 100, 255)  # Agent

WINDOW_SIZE = 500

def render_environment(env):
    if env.window is None:
        pygame.init()
        env.window = pygame.display.set_mode((env.window_size, env.window_size))
        pygame.display.set_caption("SMEEF Empowerment Simulation")
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.window.fill(WHITE)
    pygame.event.pump()  # process window events

    grid_size = env.grid_size
    cell_size = env.window_size // grid_size

    # Draw grid
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(env.window, BLACK, rect, 1)

    # Draw goal
    gx, gy = env.goal_pos
    goal_rect = pygame.Rect(gx * cell_size, gy * cell_size, cell_size, cell_size)
    pygame.draw.rect(env.window, GREEN, goal_rect)

    # Draw agent
    ax, ay = env.agent_pos
    agent_rect = pygame.Rect(ax * cell_size + 5, ay * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.rect(env.window, BLUE, agent_rect)

    # Display stats
    font = pygame.font.Font(None, 26)
    text = font.render(f"Skill: {env.skill_level} | Energy: {env.energy_level}", True, BLACK)
    env.window.blit(text, (10, 10))

    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])

    # Return frame (optional, for video recording)
    frame = pygame.surfarray.array3d(env.window)
    frame = np.transpose(frame, (1, 0, 2))
    return frame
