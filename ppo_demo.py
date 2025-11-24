"""
PPO TRAINED AGENT DEMONSTRATION
Showing the best performing model (PPO) in action
Run this from root folder to avoid import issues
"""
import pygame
import numpy as np
from stable_baselines3 import PPO
from environment.smeef_env import SMEEFEnv, Action

def demonstrate_ppo_agent(model_path, episodes=3, steps_per_episode=100):
    """Demonstrate the trained PPO agent - best performing model"""
    pygame.init()
    env = SMEEFEnv(grid_size=6, max_steps=steps_per_episode, render_mode="human")
    
    print("="*60)
    print("PPO TRAINED AGENT DEMONSTRATION")
    print("Showing best performing model (PPO) with learned behavior")
    print("="*60)
    
    try:
        # Load the trained PPO model
        model = PPO.load(model_path)
        print(f"✅ Loaded trained PPO model from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train a PPO model first or check the model path")
        return

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"\n🎯 EPISODE {ep + 1} - PPO TRAINED AGENT")
        print(f"Starting at: {env.position} | Money: ${env.resources[0]:.1f} | Energy: {env.resources[1]:.1f}")

        while not done and step_count < steps_per_episode:
            # Use trained PPO model to predict actions
            action, _states = model.predict(obs, deterministic=True)
            action_name = Action(action).name
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            done = terminated or truncated

            # Print strategic decisions (less verbose than random demo)
            if step_count % 10 == 0 or reward != 0:  # Print key moments
                print(f"Step {step_count}: {action_name:15} | Reward: {reward:6.2f} | "
                      f"Pos: {env.position} | Energy: {env.resources[1]:5.1f}")

            env.render()
            pygame.time.delay(200)  # Slightly faster than random demo

        print(f"\n📊 EPISODE {ep + 1} RESULTS - PPO PERFORMANCE:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps taken: {step_count}")
        print(f"   Final position: {env.position}")
        print(f"   Resources - Money: ${env.resources[0]:.1f}, Energy: {env.resources[1]:.1f}")
        print(f"   Skills: {env.resources[2]:.1f}, Support: {env.resources[3]:.1f}")
        print(f"   Child - Health: {env.child_status[0]:.1f}, Happiness: {env.child_status[1]:.1f}")
        
        # Analyze agent strategy
        analyze_agent_strategy(env, total_reward)

    pygame.quit()
    print("\n" + "="*60)
    print("PPO DEMONSTRATION COMPLETE")
    print("Trained agent shows strategic decision making")
    print("Compare with random actions to see learning progress")
    print("="*60)

def analyze_agent_strategy(env, total_reward):
    """Analyze and explain the PPO agent's learned strategy"""
    print(f"\n🧠 AGENT STRATEGY ANALYSIS:")
    
    # Check if agent reached goal
    if (env.position == env.goal_location).all():
        print("   ✅ SUCCESS: Agent reached the goal location!")
    else:
        print("   ❌ Agent did not reach goal (complex multi-objective task)")
    
    # Resource management analysis
    if env.resources[0] > 50:
        print("   💰 Good financial management")
    elif env.resources[0] < 10:
        print("   ⚠️  Low funds - agent struggling with money management")
    
    if env.resources[1] > 30:
        print("   🔋 Good energy conservation")
    else:
        print("   😴 Low energy - agent overworking")
    
    # Child well-being analysis
    if env.child_status[0] > 70 and env.child_status[1] > 70:
        print("   👶 Excellent child care management")
    elif env.child_status[0] < 30 or env.child_status[1] < 30:
        print("   🚨 Child needs attention")
    
    # Overall performance rating
    if total_reward > 0:
        print("   🏆 POSITIVE REWARD: Agent learning effective strategies")
    else:
        print("   📉 NEGATIVE REWARD: Environment challenging, needs more training")

def compare_random_vs_trained():
    """Quick comparison between random and trained agent performance"""
    print("\n" + "="*60)
    print("RANDOM vs TRAINED AGENT COMPARISON")
    print("="*60)
    
    # Test random agent quickly
    env = SMEEFEnv(grid_size=6, max_steps=50, render_mode=None)
    random_rewards = []
    trained_rewards = []
    
    # Random agent performance
    print("Testing random agent (5 episodes)...")
    for _ in range(5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        random_rewards.append(total_reward)
    
    # Trained agent performance (if model exists)
    try:
        model = PPO.load("models/ppo/ppo_best_model.zip")
        print("Testing trained PPO agent (5 episodes)...")
        for _ in range(5):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            trained_rewards.append(total_reward)
        
        # Comparison results
        avg_random = np.mean(random_rewards)
        avg_trained = np.mean(trained_rewards)
        improvement = ((avg_trained - avg_random) / abs(avg_random)) * 100 if avg_random != 0 else float('inf')
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Random Agent Average: {avg_random:.2f}")
        print(f"   PPO Trained Average:  {avg_trained:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print("   ✅ Training successful - PPO outperforms random actions")
        else:
            print("   ⚠️  Training needed - PPO not yet outperforming random")
            
    except Exception as e:
        print(f"❌ Cannot compare - trained model not available: {e}")
    
    env.close()

if __name__ == "__main__":
    # Try to load the best PPO model
    model_paths = [
        "models/ppo/ppo_best_model.zip",
        "models/ppo/best_model.zip", 
        "models/ppo/ppo_model.zip",
        "models/best_model.zip"
    ]
    
    model_path = None
    for path in model_paths:
        try:
            model = PPO.load(path)
            model_path = path
            break
        except:
            continue
    
    if model_path:
        demonstrate_ppo_agent(model_path)
        compare_random_vs_trained()
    else:
        print("❌ No trained PPO model found!")
        print("Please train a PPO model first using:")
        print("python main.py --train-ppo")