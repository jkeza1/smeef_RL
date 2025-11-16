import argparse
from stable_baselines3 import DQN, PPO, A2C
from environment.smeef_env import SMEEFEnv


def load_model(algo, path):
    if algo == "dqn":
        return DQN.load(path)
    elif algo == "ppo":
        return PPO.load(path)
    elif algo == "a2c":
        return A2C.load(path)
    else:
        raise ValueError("Unknown algorithm")


def train_model(algo_name):
    env = SMEEFEnv()

    if algo_name == "dqn":
        model = DQN("MlpPolicy", env, verbose=1)
    elif algo_name == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo_name == "a2c":
        model = A2C("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("Invalid algorithm")

    print(f"\nTraining {algo_name.upper()} on SMEEF Women Empowerment Grid...\n")
    model.learn(total_timesteps=5000)

    save_path = f"outputs/{algo_name}_smeef_model"
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn",
                        help="dqn, ppo, or a2c")
    args = parser.parse_args()

    train_model(args.algo)
