"""
REQUIRED: Static file showing agent taking random actions without any training
Run this from root folder to avoid import issues
"""
import pygame
import numpy as np
from environment.smeef_env import SMEEFEnv, Action

def run_random_demo(episodes=2, steps_per_episode=30):
    """Demonstrate agent taking completely random actions without any training"""
    pygame.init()
    env = SMEEFEnv(grid_size=6, max_steps=steps_per_episode, render_mode="human")
    
    print("="*60)
    print("RANDOM ACTION DEMONSTRATION")
    print("Required: Show agent taking random actions without training")
    print("="*60)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"\n🎬 EPISODE {ep + 1} - RANDOM ACTIONS ONLY")
        print(f"Starting at: {env.position} | Money: ${env.resources[0]:.1f}")

        while not done and step_count < steps_per_episode:
            # TAKE RANDOM ACTION - NO MODEL, NO TRAINING
            action = env.action_space.sample()
            action_name = Action(action).name
            
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            total_reward += reward
            done = terminated or truncated

            # Print every step to show randomness
            print(f"Step {step_count}: {action_name:15} | Reward: {reward:6.2f} | Pos: {env.position} | "
                  f"Energy: {env.resources[1]:5.1f}")

            env.render()
            pygame.time.delay(300)  # Slow down to see actions

        print(f"📊 EPISODE {ep + 1} RESULTS:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Final position: {env.position}")
        print(f"   Final resources - Money: ${env.resources[0]:.1f}, Energy: {env.resources[1]:.1f}")
        print(f"   Child status - Health: {env.child_status[0]:.1f}, Happiness: {env.child_status[1]:.1f}")

    pygame.quit()
    print("\n" + "="*60)
    print("RANDOM DEMONSTRATION COMPLETE")
    print("This shows the environment works without any RL training")
    print("="*60)

if __name__ == "__main__":
    run_random_demo()