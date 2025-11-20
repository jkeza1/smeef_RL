import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

class FastTrainingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(FastTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            try:
                # Quick evaluation with fewer episodes
                mean_reward = self.evaluate_model(n_episodes=3)
                
                # Save if best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    os.makedirs(self.save_path, exist_ok=True)
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    # Save VecNormalize stats too
                    self.training_env.save(os.path.join(self.save_path, 'vec_normalize.pkl'))
                    if self.verbose:
                        print(f"✓ New best model with mean reward: {mean_reward:.2f}")
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
    
    def evaluate_model(self, n_episodes=3):
        # Create eval env with same wrapper structure as training
        eval_env = DummyVecEnv([lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))])
        
        # CRITICAL: Sync normalization stats from training env
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        if hasattr(self.training_env, 'obs_rms'):
            eval_env.obs_rms = self.training_env.obs_rms
            eval_env.ret_rms = self.training_env.ret_rms
        
        total_reward = 0
        
        for _ in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 100  # Safety limit
            
            while steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env.step(action)
                episode_reward += reward[0]  # Extract scalar from array
                steps += 1
                
                # VecEnv returns array of dones
                if dones[0]:
                    break
            
            total_reward += episode_reward

        eval_env.close()
        return total_reward / n_episodes

def train_dqn_fast():
    print("⚡ Starting FAST DQN Training...")
    
    # Create vectorized, normalized environment with flattened observations
    make_env = lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=80))
    n_envs = 4  # Reduced to avoid overwhelming the system
    env = make_vec_env(make_env, n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)  # Add reward normalization
    
    # Create directories
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("outputs/metrics/dqn", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    
    # IMPROVED hyperparameters
    hyperparams = [
        # Baseline with better settings
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
        # Slower learning
        {'learning_rate': 0.0001, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
        # More exploration
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.7, 'exploration_final_eps': 0.02},
        # Larger buffer
        {'learning_rate': 0.0005, 'buffer_size': 50000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
        # Higher final epsilon
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.1},
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
            # Slower learning
        {'learning_rate': 0.0001, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
            # More exploration early
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.7, 'exploration_final_eps': 0.02},
            # Larger buffer to improve replay diversity
        {'learning_rate': 0.0005, 'buffer_size': 50000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.05},
            # Higher final epsilon to keep exploration
        {'learning_rate': 0.0005, 'buffer_size': 20000, 'exploration_fraction': 0.5, 'exploration_final_eps': 0.1},
    ]
    
    results = []
    best_reward = -float('inf')
    best_params = None
    
    for i, params in enumerate(hyperparams):
        print(f"\n🏃 Run {i+1}/{len(hyperparams)} - Fast Training")
        print(f"   LR: {params['learning_rate']}, Buffer: {params['buffer_size']}")
        print(f"   Explore: {params['exploration_fraction']}→{params['exploration_final_eps']}")
        
        try:
            # Reset environment
            env.reset()
            
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=params['learning_rate'],
                buffer_size=params['buffer_size'],
                learning_starts=1000,  # Start learning after some exploration
                exploration_fraction=params['exploration_fraction'],
                exploration_final_eps=params['exploration_final_eps'],
                train_freq=8,  # Less aggressive training
                gradient_steps=2,  # More updates per training step
                target_update_interval=1000,  # Update target network less frequently
                batch_size=128,  # Larger batch for stability
                gamma=0.99,  # Standard discount factor
                verbose=0,
                device="auto",
            )
            
            # Training with error handling
            total_timesteps = 100000 * n_envs  # More timesteps
            callback = FastTrainingCallback(check_freq=2000, save_path=f"models/dqn/run_{i+1}")
            
            print(f"   Training for {total_timesteps} timesteps...")
            model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
            
            # Final evaluation
            print("   Evaluating final model...")
            mean_reward = evaluate_final_model_fast(model, env)
            results.append({
                'run': i+1,
                'params': params,
                'mean_reward': mean_reward,
                'success': True
            })
            
            print(f"   ✅ Final Reward: {mean_reward:.2f}")
            
            # Save results incrementally
            np.save(f"outputs/metrics/dqn/run_{i+1}_results.npy", results[-1])
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_params = params
                model.save("models/dqn/dqn_best_model")
                env.save("models/dqn/dqn_best_vecnormalize.pkl")
                print("   💫 NEW BEST MODEL!")
                
        except Exception as e:
            print(f"   ❌ Run {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'run': i+1,
                'params': params,
                'mean_reward': -999,
                'success': False,
                'error': str(e)
            })
    
    # Save all results
    np.save("outputs/metrics/dqn/all_results.npy", results)
    
    # Plot results
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        plot_training_results_fast(successful_results)
        generate_hyperparameter_analysis(successful_results)
    
    print(f"\n🎉 FAST DQN Training Complete!")
    print(f"🏆 Best parameters: {best_params}")
    print(f"📈 Best mean reward: {best_reward:.2f}")
    
    env.close()

def evaluate_final_model_fast(model, training_env, n_episodes=10):
    """Evaluate with proper normalization"""
    eval_env = DummyVecEnv([lambda: NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Sync normalization stats
    if hasattr(training_env, 'obs_rms'):
        eval_env.obs_rms = training_env.obs_rms
        eval_env.ret_rms = training_env.ret_rms
    
    total_reward = 0
    
    for ep in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 100  # Safety limit to prevent infinite loops
        
        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            # VecEnv returns array of dones - check first element
            if dones[0]:
                break
        
        if steps >= max_steps:
            print(f"   ⚠️ Episode {ep+1} hit max steps limit")
        
        total_reward += episode_reward
    
    eval_env.close()
    return total_reward / n_episodes

def plot_training_results_fast(results):
    plt.figure(figsize=(12, 6))
    
    rewards = [r['mean_reward'] for r in results]
    runs = [r['run'] for r in results]
    
    # Color code by learning rate
    colors = []
    for result in results:
        lr = result['params']['learning_rate']
        if lr >= 0.0005:
            colors.append('#ff6b6b')  
        else:
            colors.append('#45b7d1') 
    
    bars = plt.bar(runs, rewards, alpha=0.8, color=colors)
    plt.xlabel('Training Run')
    plt.ylabel('Mean Reward')
    plt.title('DQN Fast Hyperparameter Tuning Results\n(Red=Higher LR, Blue=Lower LR)')
    plt.xticks(runs)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rewards):
        plt.text(runs[i], v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/dqn_fast_hyperparameter_results.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved to outputs/plots/dqn_fast_hyperparameter_results.png")

def generate_hyperparameter_analysis(results):
    """Generate hyperparameter analysis"""
    print(f"\n🧪 HYPERPARAMETER ANALYSIS")
    print("=" * 50)
    
    # Top 3 configurations
    top_3 = sorted(results, key=lambda x: x['mean_reward'], reverse=True)[:3]
    print(f"🏅 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(top_3):
        params = result['params']
        print(f"  {i+1}. Run {result['run']}: Reward = {result['mean_reward']:.2f}")
        print(f"     LR: {params['learning_rate']}, Buffer: {params['buffer_size']}")
        print(f"     Explore: {params['exploration_fraction']}→{params['exploration_final_eps']}")

if __name__ == "__main__":
    # Add better error handling
    try:
        train_dqn_fast()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()