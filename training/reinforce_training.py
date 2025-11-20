import os
import sys
import json
import argparse
from typing import List, Dict

import torch
import torch.optim as optim

# Make repo root importable when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environment.smeef_env import SMEEFEnv
from environment.obs_wrappers import NormalizeFlattenObs
from agents.reinforce_agent import REINFORCEAgent


def safe_reset(env):
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret


def safe_step(env, action):
    step_ret = env.step(action)
    if len(step_ret) == 5:
        obs, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = step_ret
        return obs, reward, bool(done), info


def calculate_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def run_one(config: Dict) -> Dict:
    """Run a single REINFORCE training with the provided config dict."""
    name = config.get('name', 'run')
    total_episodes = int(config.get('total_episodes', 1000))
    gamma = float(config.get('gamma', 0.99))
    lr = float(config.get('learning_rate', 1e-3))
    hidden_size = int(config.get('hidden_size', 128))
    seed = config.get('seed', None)

    if seed is not None:
        torch.manual_seed(int(seed))

    print(f"\n🚀 RUN {name}: episodes={total_episodes}, lr={lr}, gamma={gamma}, hidden={hidden_size}")

    env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, hidden_size=hidden_size, lr=lr)

    # Ensure top-level folder exists
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs(f"models/reinforce/{name}", exist_ok=True)

    episode_rewards: List[float] = []
    best_reward = float('-inf')

    for i in range(total_episodes):
        state = safe_reset(env)
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = safe_step(env, action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state

        returns = calculate_returns(rewards, gamma)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        loss = torch.stack(policy_loss).sum()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Save best model when it improves
        if total_reward > best_reward:
            best_reward = total_reward
            best_path = f"models/reinforce/policy_best_{name}.pth"
            run_best_path = f"models/reinforce/{name}/policy_best.pth"
            torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'episode': i+1, 'reward': total_reward}, best_path)
            torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'episode': i+1, 'reward': total_reward}, run_best_path)
            print(f"New best model at episode {i+1}: reward={total_reward:.2f} saved to {best_path}")

        if (i + 1) % 100 == 0:
            last = episode_rewards[-100:]
            avg = sum(last) / len(last)
            print(f"Episode {i+1}/{total_episodes} avg100={avg:.2f}")

    # Final save (top-level and per-run)
    final_top = f"models/reinforce/policy_final_{name}.pth"
    final_run = f"models/reinforce/{name}/policy_final.pth"
    torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_top)
    torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_run)
    print(f"Saved final models to {final_top} and {final_run}")

    # Compute final averaged metric
    last_n = episode_rewards[-100:] if len(episode_rewards) >= 1 else episode_rewards
    avg_last = float(sum(last_n) / len(last_n)) if len(last_n) > 0 else 0.0

    env.close()
    return {'name': name, 'best_reward': best_reward, 'final_avg100': avg_last}


def create_hyperparam_grid() -> List[Dict]:
    """Create 10 hyperparameter configurations for the sweep (adds a human-friendly group label)."""
    grid: List[Dict] = []
    names = [f'run_{i+1}' for i in range(10)]
    lrs = [5e-4, 1e-3, 3e-3]
    gammas = [0.9, 0.99, 0.999]
    hidden_sizes = [64, 128]

    i = 0
    for lr in lrs:
        for g in gammas:
            for h in hidden_sizes:
                if i >= 10:
                    break
                if lr >= 3e-3:
                    group = 'High LR'
                elif lr >= 1e-3:
                    group = 'Medium LR'
                else:
                    group = 'Low LR'

                grid.append({'name': names[i], 'group': group, 'learning_rate': lr, 'gamma': g, 'hidden_size': h, 'total_episodes': 100000})
                i += 1
            if i >= 10:
                break
        if i >= 10:
            break

    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', action='store_true', help='Run the built-in 10-config sweep')
    p.add_argument('--hp-file', type=str, default=None, help='JSON file with list of hyperparam configs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--total-episodes', type=int, default=None, help='Override episodes for each run if provided')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.dry_run:
        print('Dry run: no training will be executed')
        sys.exit(0)

    if args.sweep:
        if args.hp_file:
            with open(args.hp_file, 'r') as f:
                grid = json.load(f)
        else:
            grid = create_hyperparam_grid()

        global_best = float('-inf')
        for idx, cfg in enumerate(grid, start=1):
            if args.total_episodes is not None:
                cfg['total_episodes'] = args.total_episodes

            header_label = cfg.get('group', cfg.get('name', 'unnamed'))
            print(f"\n🏃 REINFORCE Run {idx}/{len(grid)} - {header_label}")
            print(f"   LR: {cfg.get('learning_rate')}, Steps: {cfg.get('total_episodes')}")

            res = run_one(cfg)
            best = res.get('best_reward', float('-inf'))
            print(f"   ✅ Reward: {best:.2f}")

            if best > global_best:
                global_best = best
                print("   💫 NEW BEST MODEL!")
    else:
        # Single default run
        cfg = {'name': 'default', 'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_size': 128, 'total_episodes': 100000}
        if args.total_episodes is not None:
            cfg['total_episodes'] = args.total_episodes
        print(f"\n🏃 REINFORCE Single Run - default")
        print(f"   LR: {cfg.get('learning_rate')}, Steps: {cfg.get('total_episodes')}")
        run_one(cfg)
import os
import sys
import json
import argparse
from typing import Optional, List, Dict

import torch
import torch.optim as optim

# Make repo root importable when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environment.smeef_env import SMEEFEnv
from environment.obs_wrappers import NormalizeFlattenObs
from agents.reinforce_agent import REINFORCEAgent


def safe_reset(env):
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret


def safe_step(env, action):
    step_ret = env.step(action)
    if len(step_ret) == 5:
        obs, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = step_ret
        return obs, reward, bool(done), info


def calculate_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def run_one(config: Dict):
    """Run a single REINFORCE training with the provided config dict."""
    name = config.get('name', 'run')
    total_episodes = config.get('total_episodes', 4000)
    gamma = config.get('gamma', 0.99)
    lr = config.get('learning_rate', 1e-3)
    hidden_size = config.get('hidden_size', 128)
    seed = config.get('seed', None)

    if seed is not None:
        torch.manual_seed(seed)

    print(f"\n🚀 RUN {name}: episodes={total_episodes}, lr={lr}, gamma={gamma}, hidden={hidden_size}")

    env = NormalizeFlattenObs(SMEEFEnv(grid_size=6, max_steps=50))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, hidden_size=hidden_size, lr=lr)

    # Ensure top-level reinforce folder exists for shared best/final saves
    os.makedirs("models/reinforce", exist_ok=True)
    # Per-run folder (keeps artifacts organized)
    os.makedirs(f"models/reinforce/{name}", exist_ok=True)

    episode_rewards: List[float] = []
    best_reward = float('-inf')

    for i in range(total_episodes):
        state = safe_reset(env)
        log_probs = []
        rewards = []
        episode_rewards: List[float] = []
        best_reward = float('-inf')
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = safe_step(env, action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state

        returns = calculate_returns(rewards, gamma)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        loss = torch.stack(policy_loss).sum()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Save best model when it improves
        if total_reward > best_reward:
            best_reward = total_reward
            # Save best model in top-level models/reinforce for easy discovery
            best_path = f"models/reinforce/policy_best_{name}.pth"
            torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'episode': i+1, 'reward': total_reward}, best_path)
            # Also keep a copy in the per-run folder
            run_best_path = f"models/reinforce/{name}/policy_best.pth"
            torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'episode': i+1, 'reward': total_reward}, run_best_path)
            print(f"New best model at episode {i+1}: reward={total_reward:.2f} saved to {best_path} and {run_best_path}")

        if (i + 1) % 100 == 0:
            last = episode_rewards[-100:]
            avg = sum(last) / len(last)
            print(f"Episode {i+1}/{total_episodes} avg100={avg:.2f}")

        # Save intermediate best
        # We save final model below; you can add best-model saving here if desired

    # Final save
    # Save final model to both top-level and per-run folders
    final_top = f"models/reinforce/policy_final_{name}.pth"
    final_run = f"models/reinforce/{name}/policy_final.pth"
    torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_top)
    torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_run)
    print(f"Saved final models to {final_top} and {final_run}")

    env.close()
        final_top = f"models/reinforce/policy_final_{name}.pth"
        final_run = f"models/reinforce/{name}/policy_final.pth"
        torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_top)
        torch.save({'policy_state_dict': agent.policy.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()}, final_run)
        print(f"Saved final models to {final_top} and {final_run}")

        # Compute final metrics to return
        last_n = episode_rewards[-100:] if len(episode_rewards) >= 1 else episode_rewards
        avg_last = sum(last_n) / len(last_n) if len(last_n) > 0 else 0.0

    """Create 10 hyperparameter configurations for the sweep."""
        env.close()
        return {'name': name, 'best_reward': best_reward, 'final_avg100': avg_last}
    names = [f'run_{i+1}' for i in range(10)]
    lrs = [5e-4, 1e-3, 3e-3]
    gammas = [0.9, 0.99, 0.999]
    hidden_sizes = [64, 128]

    i = 0
    for lr in lrs:
        for g in gammas:
            for h in hidden_sizes:
                if i >= 10:
                    break
                grid.append({'name': names[i], 'learning_rate': lr, 'gamma': g, 'hidden_size': h, 'total_episodes': 100000})
                i += 1
            if i >= 10:
                break
        if i >= 10:
            break

    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', action='store_true', help='Run the built-in 10-config sweep')
    p.add_argument('--hp-file', type=str, default=None, help='JSON file with list of hyperparam configs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--total-episodes', type=int, default=None, help='Override episodes for each run if provided')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.dry_run:
        print('Dry run: no training will be executed')
        exit(0)

    if args.sweep:
        if args.hp_file:
            with open(args.hp_file, 'r') as f:
                grid = json.load(f)
        else:
            grid = create_hyperparam_grid()

        global_best = float('-inf')
        for idx, cfg in enumerate(grid, start=1):
            if args.total_episodes is not None:
                cfg['total_episodes'] = args.total_episodes

            print(f"\n🏃 REINFORCE Run {idx}/{len(grid)} - {cfg.get('name','unnamed')}")
            print(f"   LR: {cfg.get('learning_rate')}, Gamma: {cfg.get('gamma')}, Hidden: {cfg.get('hidden_size')}")

            res = run_one(cfg)
            best = res.get('best_reward', float('-inf'))
            print(f"   ✅ Reward: {best:.2f}")

            if best > global_best:
                global_best = best
                print("   💫 NEW BEST MODEL!")
    else:
        # Single default run
        cfg = {'name': 'default', 'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_size': 128, 'total_episodes': 100000}
        if args.total_episodes is not None:
            cfg['total_episodes'] = args.total_episodes
        run_one(cfg)