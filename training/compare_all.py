import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.smeef_env import SMEEFEnv
import numpy as np
import matplotlib.pyplot as plt

def compare_all_algorithms():
    """Compare all trained models"""
    print("🧪 COMPARING ALL ALGORITHMS")
    print("=" * 50)
    
    algorithms = [
        ("DQN", "models/dqn/dqn_best_model"),
        ("A2C", "models/a2c/a2c_best_model"), 
        ("PPO", "models/ppo/ppo_best_model"),
        ("REINFORCE", "models/reinforce/reinforce_model", "Policy Gradient"),
    ]
    
    results = {}
    
    for algo_name, model_path in algorithms:
        if os.path.exists(model_path + ".zip"):
            reward = evaluate_model_fast(model_path, algo_name)
            results[algo_name] = reward
            print(f"📊 {algo_name}: {reward:8.2f}")
        else:
            print(f"⚠️  {algo_name}: Model not found")
            results[algo_name] = 0
    
    # Plot comparison
    plot_comparison(results)

def evaluate_model_fast(model_path, algorithm_name, n_episodes=5):
    """Fast evaluation"""
    try:
        if 'dqn' in algorithm_name.lower():
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        elif 'ppo' in algorithm_name.lower():
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        elif 'a2c' in algorithm_name.lower():
            from stable_baselines3 import A2C
            model = A2C.load(model_path)
        else:
            return 0
        
        env = SMEEFEnv(grid_size=6, max_steps=80)
        total_reward = 0
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_reward += episode_reward
        
        env.close()
        return total_reward / n_episodes
        
    except Exception as e:
        print(f"❌ Error evaluating {algorithm_name}: {e}")
        return 0

def plot_comparison(results):
    """Plot algorithm comparison"""
    algorithms = list(results.keys())
    rewards = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, rewards, alpha=0.7, 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    
    plt.ylabel('Mean Reward')
    plt.title('RL Algorithm Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(rewards):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_all_algorithms()