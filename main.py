"""
SMEEF RL - Single Mother Empowerment Environment Framework
Main entry point for running trained models and demonstrations
"""

import os
import sys
import argparse
from typing import Optional

def setup_environment():
    """Ensure all required directories exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    print("✅ Environment setup complete")

def run_random_demo():
    """Run the random action demonstration (required for assignment)"""
    print("\n🎲 Running Random Action Demonstration")
    print("=" * 50)
    try:
        from run_random_demo import run_random_demo as run_demo
        run_demo()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure run_random_demo.py exists in the root directory")

def run_trained_demo(algorithm: str = "best"):
    """Run demo with a trained model"""
    print(f"\n🤖 Running Trained Agent Demo: {algorithm.upper()}")
    print("=" * 50)
    
    # Map algorithm names to model paths
    model_paths = {
        "dqn": "models/dqn/dqn_best_model",
        "a2c": "models/a2c/a2c_best_model", 
        "ppo": "models/ppo/ppo_best_model",
        "reinforce": "models/reinforce/reinforce_model",
        "best": "models/best_overall_model"  # You can set this manually
    }
    
    model_path = model_paths.get(algorithm.lower())
    
    if not model_path or not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for algo, path in model_paths.items():
            exists = "✅" if os.path.exists(path + ".zip") else "❌"
            print(f"  {exists} {algo}: {path}")
        return
    
    try:
        # Update smeef_demo.py to use the selected model
        import subprocess
        env = os.environ.copy()
        env['SMEEF_ALGORITHM'] = algorithm
        env['SMEEF_MODEL_PATH'] = model_path
        
        subprocess.run([sys.executable, "smeef_demo.py"], env=env)
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def run_hyperparameter_tuning(algorithm: str):
    """Run hyperparameter tuning for a specific algorithm"""
    print(f"\n🧪 Running Hyperparameter Tuning: {algorithm.upper()}")
    print("=" * 50)
    
    tuning_scripts = {
        "dqn": "training.dqn_training.train_dqn_ultra_fast",
        "a2c": "training.a2c_training.train_a2c_ultra_fast",
        "ppo": "training.ppo_training.train_ppo_ultra_fast",
        "reinforce": "training.reinforce_training.train_reinforce",
        "all": "training.hyperparameter_sweep.run_all_tuning"
    }
    
    script = tuning_scripts.get(algorithm.lower())
    if not script:
        print(f"❌ Unknown algorithm: {algorithm}")
        print("Available: dqn, a2c, ppo, reinforce, all")
        return
    
    try:
        module_name, function_name = script.rsplit('.', 1)
        module = __import__(module_name, fromlist=[function_name])
        function = getattr(module, function_name)
        function()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure the training script exists")

def run_algorithm_comparison():
    """Run comprehensive algorithm comparison"""
    print("\n📊 Running Algorithm Comparison")
    print("=" * 50)
    try:
        from training.final_comparison import comprehensive_comparison
        comprehensive_comparison()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure training/final_comparison.py exists")

def show_environment_info():
    """Display information about the environment"""
    print("\n🏠 SMEEF Environment Information")
    print("=" * 50)
    
    try:
        from environment.smeef_env import SMEEFEnv, Action
        
        env = SMEEFEnv(grid_size=6, max_steps=100)
        
        print("Action Space:")
        for action in Action:
            print(f"  {action.name}: {action.value}")
        
        print(f"\nObservation Space: {env.observation_space}")
        print(f"Grid Size: {env.grid_size}")
        print(f"Max Steps: {env.max_steps}")
        
        print("\nService Locations:")
        for service_name, service_info in env.services.items():
            print(f"  {service_name}: {service_info['positions']}")
        
        env.close()
        
    except ImportError as e:
        print(f"❌ Error: {e}")

def generate_report():
    """Generate a summary report of the project"""
    print("\n📋 Generating Project Report")
    print("=" * 50)
    
    report = f"""
    SMEEF RL PROJECT REPORT
    ======================
    
    Project Structure:
    - environment/: Custom Gymnasium environment
    - training/: RL algorithm implementations
    - models/: Trained model files
    - outputs/: Results, plots, and metrics
    - main.py: Project entry point
    
    Implemented Algorithms:
    - DQN (Deep Q-Network): Value-based method
    - A2C (Advantage Actor-Critic): Policy gradient with value function
    - PPO (Proximal Policy Optimization): Policy gradient with clipping
    - REINFORCE: Policy gradient method
    
    Key Features:
    - Custom 6x6 grid environment
    - 8 discrete actions
    - Dict observation space with multiple components
    - Balanced reward structure
    - Hyperparameter tuning for all algorithms
    
    Usage:
    - Random demo: python main.py --random
    - Trained demo: python main.py --demo [algorithm]
    - Hyperparameter tuning: python main.py --tune [algorithm]
    - Algorithm comparison: python main.py --compare
    """
    
    print(report)
    
    # Save report to file
    with open("outputs/project_report.txt", "w") as f:
        f.write(report)
    print("✅ Report saved to outputs/project_report.txt")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="SMEEF RL - Single Mother Empowerment Environment")
    
    parser.add_argument("--setup", action="store_true", help="Setup environment directories")
    parser.add_argument("--random", action="store_true", help="Run random action demonstration")
    parser.add_argument("--demo", type=str, help="Run demo with trained model (dqn, a2c, ppo, reinforce, best)")
    parser.add_argument("--tune", type=str, help="Run hyperparameter tuning (dqn, a2c, ppo, reinforce, all)")
    parser.add_argument("--compare", action="store_true", help="Run algorithm comparison")
    parser.add_argument("--info", action="store_true", help="Show environment information")
    parser.add_argument("--report", action="store_true", help="Generate project report")
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*60)
    print("          SMEEF RL - Single Mother Empowerment")
    print("="*60)
    
    # Execute requested action
    if args.setup:
        setup_environment()
    elif args.random:
        run_random_demo()
    elif args.demo:
        run_trained_demo(args.demo)
    elif args.tune:
        run_hyperparameter_tuning(args.tune)
    elif args.compare:
        run_algorithm_comparison()
    elif args.info:
        show_environment_info()
    elif args.report:
        generate_report()
    else:
        # Interactive mode
        print("\n🚀 Welcome to SMEEF RL Project!")
        print("Choose an option:")
        print("1. Run Random Action Demo (Required for assignment)")
        print("2. Run Trained Agent Demo")
        print("3. Run Hyperparameter Tuning") 
        print("4. Run Algorithm Comparison")
        print("5. Show Environment Info")
        print("6. Generate Project Report")
        print("7. Exit")
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            if choice == "1":
                run_random_demo()
            elif choice == "2":
                algo = input("Enter algorithm (dqn/a2c/ppo/reinforce/best): ").strip()
                run_trained_demo(algo)
            elif choice == "3":
                algo = input("Enter algorithm (dqn/a2c/ppo/reinforce/all): ").strip()
                run_hyperparameter_tuning(algo)
            elif choice == "4":
                run_algorithm_comparison()
            elif choice == "5":
                show_environment_info()
            elif choice == "6":
                generate_report()
            elif choice == "7":
                print("👋 Goodbye!")
                return
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

if __name__ == "__main__":
    main()