import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from environment.smeef_env import SMEEFEnv
from environment.obs_wrappers import NormalizeFlattenObs


def safe_reset(env):
    """Return observation from env.reset() and ignore optional info dict."""
    reset_ret = env.reset()
    if isinstance(reset_ret, tuple):
        return reset_ret[0]
    return reset_ret


def safe_step(env, action):
    """Handle both old (obs, reward, done, info) and new (obs, reward, terminated, truncated, info) APIs.
    Returns (obs, reward, done, info)
    """
    step_ret = env.step(action)
    if len(step_ret) == 5:
        obs, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = step_ret
        return obs, reward, bool(done), info


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


def ensure_dirs():
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("outputs/logs/a2c", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)


def evaluate_a2c_ultra_fast(model, n_episodes=5):
    """Ultra-fast evaluation for A2C using the flattened wrapper."""
    eval_env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))
    total_reward = 0.0

    for _ in range(n_episodes):
        obs = safe_reset(eval_env)
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = safe_step(eval_env, action)
            episode_reward += float(reward)

        total_reward += episode_reward

    try:
        eval_env.close()
    except Exception:
        pass

    return total_reward / float(n_episodes)


def analyze_a2c_results_fast(results, best_params, best_reward):
    """Quick analysis for A2C results"""
    plt.figure(figsize=(12, 6))
    groups = [r['group'] for r in results]
    rewards = [r['mean_reward'] for r in results]

    colors = []
    for result in results:
        lr = result['learning_rate']
        if lr >= 0.001:
            colors.append('#ff6b6b')
        elif lr >= 0.0005:
            colors.append('#4ecdc4')
        else:
            colors.append('#45b7d1')

    bars = plt.bar(range(len(results)), rewards, color=colors, alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Mean Reward')
    plt.title('A2C Ultra-Fast Hyperparameter Tuning')
    plt.xticks(range(len(results)), groups, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    for i, v in enumerate(rewards):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/plots/a2c_ultra_fast_results.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n🧪 A2C QUICK ANALYSIS")
    print("=" * 40)
    top_3 = sorted(results, key=lambda x: x['mean_reward'], reverse=True)[:3]
    print("🏅 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(top_3):
        print(f"  {i+1}. {result['group']}: {result['mean_reward']:.2f}")

    lr_performance = {}
    for result in results:
        lr = result['learning_rate']
        lr_performance.setdefault(lr, []).append(result['mean_reward'])

    print("\n📊 Learning Rate Performance:")
    for lr, rewards in sorted(lr_performance.items()):
        avg = np.mean(rewards)
        print(f"  LR {lr}: {avg:.2f}")


def train_a2c_ultra_fast():
    print("⚡ Starting ULTRA-FAST A2C Hyperparameter Tuning...")
    ensure_dirs()

    # Use flattened observations so policy is an MLP
    base_env = lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=60))
    env = base_env()
    env = Monitor(env, "outputs/logs/a2c/")

    hyperparams = create_a2c_hyperparameter_combinations()
    results = []
    best_reward = -float('inf')
    best_params = None

    for i, params in enumerate(hyperparams):
        print(f"\n🏃 A2C Run {i+1}/{len(hyperparams)} - {params['group']}")
        print(f"   LR: {params['learning_rate']}, Steps: {params['n_steps']}")

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            verbose=0,
            device="auto",
        )

        model.learn(total_timesteps=2000)

        mean_reward = evaluate_a2c_ultra_fast(model)

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

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
            model.save("models/a2c/a2c_best_model")
            print("   💫 NEW BEST MODEL!")

    analyze_a2c_results_fast(results, best_params, best_reward)

    print(f"\n🎉 ULTRA-FAST A2C Complete!")
    print(f"🏆 Best Model: {best_params['group'] if best_params is not None else 'none'}")
    print(f"📈 Best Mean Reward: {best_reward:.2f}")

    try:
        env.close()
    except Exception:
        pass

    return results


if __name__ == "__main__":
    train_a2c_ultra_fast()
