import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from environment.smeef_env import SMEEFEnv

def create_a2c_hyperparameter_combinations():
    """Generate 10 different A2C hyperparameter combinations"""
    return [
        # Learning Rate Variations
        {'learning_rate': 0.001, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'High LR'},
        {'learning_rate': 0.0007, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Medium LR'},
        {'learning_rate': 0.0003, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Low LR'},
        
        # Step Variations
        {'learning_rate': 0.0007, 'n_steps': 32, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Short Steps'},
        {'learning_rate': 0.0007, 'n_steps': 128, 'ent_coef': 0.01, 'vf_coef': 0.5, 'group': 'Long Steps'},
        
        # Entropy Coefficient Variations
        {'learning_rate': 0.0007, 'n_steps': 64, 'ent_coef': 0.001, 'vf_coef': 0.5, 'group': 'Low Entropy'},
        {'learning_rate': 0.0007, 'n_steps': 64, 'ent_coef': 0.1, 'vf_coef': 0.5, 'group': 'High Entropy'},
        
        # Value Function Coefficient Variations
        {'learning_rate': 0.0007, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.25, 'group': 'Low VF Coef'},
        {'learning_rate': 0.0007, 'n_steps': 64, 'ent_coef': 0.01, 'vf_coef': 0.75, 'group': 'High VF Coef'},
        
        # Combined
        {'learning_rate': 0.001, 'n_steps': 128, 'ent_coef': 0.1, 'vf_coef': 0.75, 'group': 'Aggressive'},
    ]

def train_a2c_ultra_fast():
    print("⚡ Starting ULTRA-FAST A2C Hyperparameter Tuning...")
    
    # Create environment with very small settings
    env = SMEEFEnv(grid_size=6, max_steps=60)  # Reduced from 80 to 60
    env = Monitor(env, "outputs/logs/a2c/")
    
    # Create directories
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("outputs/metrics/a2c", exist_ok=True)
    
    # Hyperparameter combinations (10 runs)
    hyperparams = create_a2c_hyperparameter_combinations()
    results = []
    
    best_reward = -float('inf')
    best_params = None
    
    for i, params in enumerate(hyperparams):
        print(f"\n🏃 A2C Run {i+1}/10 - {params['group']}")
        print(f"   LR: {params['learning_rate']}, Steps: {params['n_steps']}")
        
        # Create A2C model with current hyperparameters
        model = A2C(
            "MultiInputPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],  # Smaller steps for faster training
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            verbose=0,
        )
        
        # ULTRA-FAST training - only 2000 steps!
        model.learn(total_timesteps=2000)  # Reduced from 6000 to 2000
        
        # Quick evaluation
        mean_reward = evaluate_a2c_ultra_fast(model)
        
        # Store results
        result = {
            'run_id': i + 1,
            'group': params['group'],
            'learning_rate': params['learning_rate'],
            'n_steps': params['n_steps'],
            'ent_coef': params['ent_coef'],
            'vf_coef': params['vf_coef'],
            'mean_reward': mean_reward,
        }
        
        results.append(result)
        
        print(f"   ✅ Reward: {mean_reward:6.2f}")
        
        # Save if best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
            model.save("models/a2c/a2c_best_model")
            print("   💫 NEW BEST MODEL!")
    
    # Quick analysis
    analyze_a2c_results_fast(results, best_params, best_reward)
    
    print(f"\n🎉 ULTRA-FAST A2C Complete! (~1-2 minutes)")
    print(f"🏆 Best Model: {best_params['group']}")
    print(f"📈 Best Mean Reward: {best_reward:.2f}")
    
    env.close()
    return results

def evaluate_a2c_ultra_fast(model, n_episodes=5):  # Reduced from 8 to 5
    """Ultra-fast evaluation for A2C"""
    eval_env = SMEEFEnv(grid_size=6, max_steps=60)  # Reduced from 80 to 60
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
    
    eval_env.close()
    return total_reward / n_episodes

def analyze_a2c_results_fast(results, best_params, best_reward):
    """Quick analysis for A2C results"""
    
    # Simple bar plot
    plt.figure(figsize=(12, 6))
    
    groups = [r['group'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    
    # Color by learning rate
    colors = []
    for result in results:
        lr = result['learning_rate']
        if lr >= 0.001:
            colors.append('#ff6b6b')  # Red for high LR
        elif lr >= 0.0005:
            colors.append('#4ecdc4')  # Teal for medium LR
        else:
            colors.append('#45b7d1')  # Blue for low LR
    
    bars = plt.bar(range(len(results)), rewards, color=colors, alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Mean Reward')
    plt.title('A2C Ultra-Fast Hyperparameter Tuning\n(Red=High LR, Teal=Medium LR, Blue=Low LR)')
    plt.xticks(range(len(results)), groups, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(rewards):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/a2c_ultra_fast_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Quick text analysis
    print(f"\n🧪 A2C QUICK ANALYSIS")
    print("=" * 40)
    
    # Top 3
    top_3 = sorted(results, key=lambda x: x['mean_reward'], reverse=True)[:3]
    print("🏅 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(top_3):
        print(f"  {i+1}. {result['group']}: {result['mean_reward']:.2f}")
    
    # Learning rate analysis
    lr_performance = {}
    for result in results:
        lr = result['learning_rate']
        if lr not in lr_performance:
            lr_performance[lr] = []
        lr_performance[lr].append(result['mean_reward'])
    
    print("\n📊 Learning Rate Performance:")
    for lr, rewards in sorted(lr_performance.items()):
        avg = np.mean(rewards)
        print(f"  LR {lr}: {avg:.2f}")

# Even faster - single run for instant testing
def train_a2c_instant():
    """Instant A2C test - completes in ~15 seconds"""
    print("⚡ Starting INSTANT A2C Test...")
    
    env = SMEEFEnv(grid_size=6, max_steps=50)
    
    # Single optimized configuration
    model = A2C(
        "MultiInputPolicy",
        env,
        learning_rate=0.0007,
        n_steps=64,  # Smaller for speed
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
    )
    
    # Ultra-short training - only 1000 steps!
    print("Training for 1000 steps...")
    model.learn(total_timesteps=1000)
    model.save("models/a2c/a2c_instant_model")
    
    # Instant evaluation
    mean_reward = evaluate_a2c_ultra_fast(model, n_episodes=3)  # Only 3 episodes
    print(f"✅ Instant A2C complete! Mean reward: {mean_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Choose one:
    train_a2c_ultra_fast()      
    