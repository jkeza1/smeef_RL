import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.smeef_env import SMEEFEnv

def create_standardized_evaluation():
    """Create identical evaluation conditions for all models"""
    eval_env_config = {
        'grid_size': 6,
        'max_steps': 100,
        'render_mode': None
    }
    
    evaluation_protocol = {
        'n_episodes': 10,
        'deterministic': True,
        'max_steps_per_episode': 100
    }
    
    return eval_env_config, evaluation_protocol

def load_training_metrics():
    """Load training progress metrics from all algorithms"""
    metrics = {}
    
    try:
        # DQN metrics
        dqn_metrics = np.load('outputs/metrics/dqn/training_metrics.npy', allow_pickle=True).item()
        metrics['DQN'] = {
            'convergence_episode': estimate_convergence(dqn_metrics.get('episode_rewards', [])),
            'avg_episode_length': dqn_metrics.get('avg_episode_length', 50),
            'training_time': dqn_metrics.get('training_time', 0),
            'hyperparameters_tested': len(dqn_metrics.get('results', [])),
            'best_hyperparams': dqn_metrics.get('best_config', {})
        }
    except:
        metrics['DQN'] = {'convergence_episode': 0, 'avg_episode_length': 0, 'training_time': 0, 'hyperparameters_tested': 0}
    
    try:
        # PPO metrics
        ppo_metrics = np.load('outputs/metrics/ppo/training_metrics.npy', allow_pickle=True).item()
        metrics['PPO'] = {
            'convergence_episode': estimate_convergence(ppo_metrics.get('episode_rewards', [])),
            'avg_episode_length': ppo_metrics.get('avg_episode_length', 50),
            'training_time': ppo_metrics.get('training_time', 0),
            'hyperparameters_tested': len(ppo_metrics.get('results', [])),
            'best_hyperparams': ppo_metrics.get('best_config', {})
        }
    except:
        metrics['PPO'] = {'convergence_episode': 0, 'avg_episode_length': 0, 'training_time': 0, 'hyperparameters_tested': 0}
    
    try:
        # A2C metrics
        a2c_metrics = np.load('outputs/metrics/a2c/training_metrics.npy', allow_pickle=True).item()
        metrics['A2C'] = {
            'convergence_episode': estimate_convergence(a2c_metrics.get('episode_rewards', [])),
            'avg_episode_length': a2c_metrics.get('avg_episode_length', 50),
            'training_time': a2c_metrics.get('training_time', 0),
            'hyperparameters_tested': len(a2c_metrics.get('results', [])),
            'best_hyperparams': a2c_metrics.get('best_config', {})
        }
    except:
        metrics['A2C'] = {'convergence_episode': 0, 'avg_episode_length': 0, 'training_time': 0, 'hyperparameters_tested': 0}
    
    try:
        # REINFORCE metrics
        reinforce_results = np.load('outputs/metrics/reinforce/training_results.npy', allow_pickle=True)
        if reinforce_results:
            rewards = [r.get('best_reward', 0) for r in reinforce_results]
            metrics['REINFORCE'] = {
                'convergence_episode': estimate_convergence(rewards),
                'avg_episode_length': 60,  # Default for REINFORCE
                'training_time': 0,
                'hyperparameters_tested': len(reinforce_results),
                'best_hyperparams': reinforce_results[0] if reinforce_results else {}
            }
    except:
        metrics['REINFORCE'] = {'convergence_episode': 0, 'avg_episode_length': 0, 'training_time': 0, 'hyperparameters_tested': 0}
    
    return metrics

def estimate_convergence(rewards, window=10, threshold=0.95):
    """Estimate when the algorithm converged based on reward stability"""
    if len(rewards) < window:
        return len(rewards)
    
    for i in range(len(rewards) - window):
        window_rewards = rewards[i:i+window]
        if max(window_rewards) > 0:  # Avoid division by zero
            stability = min(window_rewards) / max(window_rewards)
            if stability >= threshold:
                return i + window
    
    return len(rewards)

def analyze_hyperparameters():
    """Analyze hyperparameter performance across all algorithms"""
    hyperparam_analysis = {}
    
    try:
        # DQN hyperparameters
        dqn_metrics = np.load('outputs/metrics/dqn/training_metrics.npy', allow_pickle=True).item()
        hyperparam_analysis['DQN'] = {
            'best_learning_rate': dqn_metrics.get('best_config', {}).get('learning_rate', 0),
            'best_exploration': dqn_metrics.get('best_config', {}).get('exploration_final_eps', 0),
            'performance_range': calculate_performance_range(dqn_metrics.get('results', [])),
            'optimal_config': extract_optimal_config(dqn_metrics.get('best_config', {}))
        }
    except:
        hyperparam_analysis['DQN'] = {}
    
    try:
        # PPO hyperparameters
        ppo_metrics = np.load('outputs/metrics/ppo/training_metrics.npy', allow_pickle=True).item()
        hyperparam_analysis['PPO'] = {
            'best_learning_rate': ppo_metrics.get('best_config', {}).get('learning_rate', 0),
            'best_clip_range': ppo_metrics.get('best_config', {}).get('clip_range', 0),
            'performance_range': calculate_performance_range(ppo_metrics.get('results', [])),
            'optimal_config': extract_optimal_config(ppo_metrics.get('best_config', {}))
        }
    except:
        hyperparam_analysis['PPO'] = {}
    
    try:
        # A2C hyperparameters
        a2c_metrics = np.load('outputs/metrics/a2c/training_metrics.npy', allow_pickle=True).item()
        hyperparam_analysis['A2C'] = {
            'best_learning_rate': a2c_metrics.get('best_config', {}).get('learning_rate', 0),
            'best_entropy_coef': a2c_metrics.get('best_config', {}).get('ent_coef', 0),
            'performance_range': calculate_performance_range(a2c_metrics.get('results', [])),
            'optimal_config': extract_optimal_config(a2c_metrics.get('best_config', {}))
        }
    except:
        hyperparam_analysis['A2C'] = {}
    
    try:
        # REINFORCE hyperparameters
        reinforce_results = np.load('outputs/metrics/reinforce/training_results.npy', allow_pickle=True)
        if reinforce_results:
            best_run = max(reinforce_results, key=lambda x: x.get('best_reward', 0))
            hyperparam_analysis['REINFORCE'] = {
                'best_learning_rate': best_run.get('learning_rate', 0),
                'best_gamma': best_run.get('gamma', 0),
                'performance_range': max([r.get('best_reward', 0) for r in reinforce_results]) - min([r.get('best_reward', 0) for r in reinforce_results]),
                'optimal_config': extract_optimal_config(best_run)
            }
    except:
        hyperparam_analysis['REINFORCE'] = {}
    
    return hyperparam_analysis

def calculate_performance_range(results):
    """Calculate performance range across hyperparameter runs"""
    if not results:
        return 0
    rewards = [r.get('mean_reward', 0) for r in results if isinstance(r, dict)]
    return max(rewards) - min(rewards) if rewards else 0

def extract_optimal_config(config):
    """Extract key hyperparameters from config"""
    optimal = {}
    for key, value in config.items():
        if key not in ['mean_reward', 'group', 'run_id']:
            optimal[key] = float(value) if isinstance(value, (int, float)) else value
    return optimal

def evaluate_all_models_standardized():
    """Evaluate all best models under identical conditions"""
    print("⚖️ FAIR MODEL COMPARISON - Standardized Evaluation")
    print("="*60)
    
    eval_config, eval_protocol = create_standardized_evaluation()
    
    models_to_evaluate = [
        ("DQN", "models/dqn/dqn_best_model.zip", load_sb3_model),
        ("PPO", "models/ppo/ppo_best_model.zip", load_sb3_model),
        ("A2C", "models/a2c/a2c_best_model.zip", load_sb3_model),
        ("REINFORCE", "models/reinforce/default/policy_best.pth", load_reinforce_model),
    ]
    
    results = {}
    
    for algo_name, model_path, load_func in tqdm(models_to_evaluate, desc="Evaluating models"):
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            results[algo_name] = 0
            continue
            
        try:
            model = load_func(model_path, algo_name)
            mean_reward = evaluate_model_standardized(
                model, algo_name, eval_config, eval_protocol
            )
            
            results[algo_name] = mean_reward
            print(f"✅ {algo_name:12}: {mean_reward:7.2f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {algo_name}: {e}")
            results[algo_name] = 0
    
    return results

def load_sb3_model(model_path, algo_name):
    """Load Stable Baselines 3 models"""
    if 'dqn' in algo_name.lower():
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif 'ppo' in algo_name.lower():
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif 'a2c' in algo_name.lower():
        from stable_baselines3 import A2C
        return A2C.load(model_path)

def load_reinforce_model(model_path, algo_name):
    """Load REINFORCE model"""
    import torch
    from agents.reinforce_agent import REINFORCEAgent
    
    checkpoint = torch.load(model_path, map_location="cpu")
    obs_dim = 12
    act_dim = 8
    
    agent = REINFORCEAgent(obs_dim, act_dim)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy.eval()
    
    return agent

def evaluate_model_standardized(model, algo_name, eval_config, eval_protocol):
    """Evaluate a single model under standardized conditions"""
    env = SMEEFEnv(**eval_config)
    total_reward = 0
    
    for episode in range(eval_protocol['n_episodes']):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < eval_protocol['max_steps_per_episode']:
            if algo_name == "REINFORCE":
                action = get_reinforce_action(model, obs)
            else:
                action = get_sb3_action(model, obs, eval_protocol['deterministic'])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    env.close()
    return total_reward / eval_protocol['n_episodes']

def get_sb3_action(model, obs, deterministic):
    """Get action from SB3 model"""
    return model.predict(obs, deterministic=deterministic)[0]

def get_reinforce_action(model, obs):
    """Get action from REINFORCE model (deterministic)"""
    import torch
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        probs = model.policy(obs_tensor)
        return torch.argmax(probs).item()

def analyze_exploration_behavior():
    """Analyze exploration vs exploitation strategies"""
    return {
        'DQN': {
            'strategy': 'Epsilon-Greedy with Decay',
            'exploration_strength': 'High early, decreases over time',
            'exploitation_strength': 'Gradually increases',
            'suitable_for': 'Discrete action spaces, stable Q-learning',
            'weaknesses': 'May get stuck in local optima, sensitive to hyperparameters'
        },
        'PPO': {
            'strategy': 'Policy Entropy with Clipped Objectives',
            'exploration_strength': 'Consistent through entropy bonus',
            'exploitation_strength': 'Strong with clipped policy updates',
            'suitable_for': 'Both discrete and continuous actions',
            'weaknesses': 'Complex hyperparameter tuning, computationally intensive'
        },
        'A2C': {
            'strategy': 'Advantage-Based Policy Updates',
            'exploration_strength': 'Moderate through stochastic policy',
            'exploitation_strength': 'Strong with advantage estimation',
            'suitable_for': 'Faster training than PPO',
            'weaknesses': 'Higher variance, less stable than PPO'
        },
        'REINFORCE': {
            'strategy': 'Monte Carlo Policy Gradient',
            'exploration_strength': 'High variance, episodic',
            'exploitation_strength': 'Weak due to high variance',
            'suitable_for': 'Simple implementations, educational purposes',
            'weaknesses': 'Slow convergence, high variance, inefficient'
        }
    }

def plot_comprehensive_comparison(performance_results, training_metrics, hyperparam_analysis):
    """Create comprehensive comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance Comparison
    algorithms = list(performance_results.keys())
    rewards = list(performance_results.values())
    
    bars1 = ax1.bar(algorithms, rewards, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], alpha=0.8)
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Algorithm Performance Comparison\n(Higher is Better)')
    ax1.grid(True, alpha=0.3)
    
    for i, (bar, reward) in enumerate(zip(bars1, rewards)):
        ax1.text(bar.get_x() + bar.get_width()/2., reward + 0.5, f'{reward:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Training Efficiency
    convergence_data = [training_metrics[algo].get('convergence_episode', 0) for algo in algorithms]
    bars2 = ax2.bar(algorithms, convergence_data, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], alpha=0.8)
    ax2.set_ylabel('Convergence Episode')
    ax2.set_title('Training Efficiency\n(Lower is Better)')
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, conv) in enumerate(zip(bars2, convergence_data)):
        ax2.text(bar.get_x() + bar.get_width()/2., conv + 5, f'{conv}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Hyperparameter Sensitivity
    sensitivity_data = [hyperparam_analysis[algo].get('performance_range', 0) for algo in algorithms]
    bars3 = ax3.bar(algorithms, sensitivity_data, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], alpha=0.8)
    ax3.set_ylabel('Performance Range')
    ax3.set_title('Hyperparameter Sensitivity\n(Lower is More Robust)')
    ax3.grid(True, alpha=0.3)
    
    for i, (bar, sens) in enumerate(zip(bars3, sensitivity_data)):
        ax3.text(bar.get_x() + bar.get_width()/2., sens + 0.1, f'{sens:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Episode Length Distribution
    episode_lengths = [training_metrics[algo].get('avg_episode_length', 0) for algo in algorithms]
    bars4 = ax4.bar(algorithms, episode_lengths, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], alpha=0.8)
    ax4.set_ylabel('Average Episode Length')
    ax4.set_title('Behavior Characteristics\n(Indicates Strategy)')
    ax4.grid(True, alpha=0.3)
    
    for i, (bar, length) in enumerate(zip(bars4, episode_lengths)):
        ax4.text(bar.get_x() + bar.get_width()/2., length + 1, f'{length}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig('outputs/plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(performance_results, training_metrics, hyperparam_analysis, exploration_analysis):
    """Generate a comprehensive report for PDF export"""
    report = {
        'executive_summary': {
            'best_performer': max(performance_results, key=performance_results.get),
            'worst_performer': min(performance_results, key=performance_results.get),
            'performance_gap': performance_results[max(performance_results, key=performance_results.get)] - performance_results[min(performance_results, key=performance_results.get)],
            'most_efficient': min(training_metrics, key=lambda x: training_metrics[x].get('convergence_episode', float('inf'))),
            'most_robust': min(hyperparam_analysis, key=lambda x: hyperparam_analysis[x].get('performance_range', float('inf')))
        },
        'performance_comparison': performance_results,
        'training_efficiency': training_metrics,
        'hyperparameter_analysis': hyperparam_analysis,
        'exploration_analysis': exploration_analysis,
        'recommendations': generate_recommendations(performance_results, training_metrics, hyperparam_analysis),
        'methodology': {
            'evaluation_episodes': 10,
            'environment': 'SMEEFEnv (6x6 grid, 100 max steps)',
            'evaluation_mode': 'Deterministic',
            'metrics_captured': ['Mean Reward', 'Convergence Episode', 'Hyperparameter Sensitivity', 'Episode Length']
        }
    }
    
    # Save report as JSON
    with open('outputs/comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def generate_recommendations(performance, training_metrics, hyperparam_analysis):
    """Generate practical recommendations based on analysis"""
    best_algo = max(performance, key=performance.get)
    
    recommendations = {
        'best_overall': f"{best_algo} - Highest performance with reasonable training efficiency",
        'for_beginners': "PPO - Most stable and well-documented",
        'for_computation': "A2C - Faster training than PPO with good performance", 
        'for_simple_impl': "REINFORCE - Educational purposes, understand policy gradients",
        'hyperparameter_advice': "Focus on learning rate and exploration parameters first",
        'environment_specific': f"For SMEEF environment, {best_algo} handles multi-objective optimization best"
    }
    
    return recommendations

def print_detailed_analysis(report):
    """Print comprehensive analysis to console"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RL ALGORITHM ANALYSIS REPORT")
    print("="*80)
    
    # Executive Summary
    summary = report['executive_summary']
    print(f"\n🏆 EXECUTIVE SUMMARY")
    print(f"   Best Performer: {summary['best_performer']}")
    print(f"   Worst Performer: {summary['worst_performer']}")
    print(f"   Performance Gap: {summary['performance_gap']:.2f}")
    print(f"   Most Efficient: {summary['most_efficient']}")
    print(f"   Most Robust: {summary['most_robust']}")
    
    # Performance Ranking
    print(f"\n📈 PERFORMANCE RANKING")
    ranked = sorted(report['performance_comparison'].items(), key=lambda x: x[1], reverse=True)
    for i, (algo, reward) in enumerate(ranked, 1):
        print(f"   {i}. {algo:12}: {reward:7.2f}")
    
    # Training Efficiency
    print(f"\n⚡ TRAINING EFFICIENCY")
    for algo, metrics in report['training_efficiency'].items():
        print(f"   {algo:12}: Converged in {metrics.get('convergence_episode', 'N/A'):4} episodes")
    
    # Exploration Analysis
    print(f"\n🎯 EXPLORATION STRATEGIES")
    for algo, strategy in report['exploration_analysis'].items():
        print(f"   {algo:12}: {strategy['strategy']}")
        print(f"               Strengths: {strategy['suitable_for']}")
        print(f"               Weaknesses: {strategy['weaknesses']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    for key, advice in report['recommendations'].items():
        if key != 'environment_specific':  # Save this for last
            print(f"   • {advice}")
    
    print(f"   • {report['recommendations']['environment_specific']}")
    
    print(f"\n📋 METHODOLOGY")
    print(f"   Environment: {report['methodology']['environment']}")
    print(f"   Evaluation: {report['methodology']['evaluation_episodes']} episodes, {report['methodology']['evaluation_mode']} mode")
    print(f"   Metrics: {', '.join(report['methodology']['metrics_captured'])}")

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE RL ALGORITHM COMPARISON")
    print("="*60)
    
    # Step 1: Standardized performance evaluation
    print("\n1. 🔄 Running Standardized Evaluation...")
    performance_results = evaluate_all_models_standardized()
    
    # Step 2: Load training metrics
    print("\n2. 📊 Loading Training Metrics...")
    training_metrics = load_training_metrics()
    
    # Step 3: Analyze hyperparameters
    print("\n3. 🎛️  Analyzing Hyperparameter Performance...")
    hyperparam_analysis = analyze_hyperparameters()
    
    # Step 4: Analyze exploration behavior
    print("\n4. 🧭 Analyzing Exploration Strategies...")
    exploration_analysis = analyze_exploration_behavior()
    
    # Step 5: Generate comprehensive plots
    print("\n5. 📈 Generating Comprehensive Visualizations...")
    plot_comprehensive_comparison(performance_results, training_metrics, hyperparam_analysis)
    
    # Step 6: Generate full report
    print("\n6. 📋 Generating Comprehensive Report...")
    report = generate_comprehensive_report(
        performance_results, training_metrics, hyperparam_analysis, exploration_analysis
    )
    
    # Step 7: Print detailed analysis
    print_detailed_analysis(report)
    
    # Step 8: Save summary for PDF report
    print("\n7. 💾 Saving Results...")
    np.save('outputs/final_comparison_results.npy', {
        'performance': performance_results,
        'training_metrics': training_metrics,
        'hyperparam_analysis': hyperparam_analysis,
        'exploration_analysis': exploration_analysis
    })
    
    print(f"\n✅ COMPARISON COMPLETE!")
    print(f"📁 Results saved to:")
    print(f"   - outputs/plots/comprehensive_comparison.png")
    print(f"   - outputs/comprehensive_report.json") 
    print(f"   - outputs/final_comparison_results.npy")
    
    return report

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Run comprehensive comparison
    final_report = main()