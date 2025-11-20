import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeFlattenObs(gym.ObservationWrapper):
    """Convert the Dict observation from SMEEFEnv into a flat normalized vector.

    Output vector layout (length 12):
      [pos_x_norm, pos_y_norm, money, energy, skills, support,
       childcare_need, financial_need, emotional_need, career_need,
       child_health, child_happiness]

    Normalization: positions divided by (grid_size-1), others divided by 100.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Expect env to have grid_size attribute
        grid_size = getattr(env, "grid_size", 8)
        # New observation space
        low = np.zeros(12, dtype=np.float32)
        high = np.ones(12, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        # obs is expected to be a dict-like object
        pos = np.asarray(obs['position'], dtype=np.float32)
        resources = np.asarray(obs['resources'], dtype=np.float32)
        needs = np.asarray(obs['needs'], dtype=np.float32)
        child = np.asarray(obs['child_status'], dtype=np.float32)

        # Normalize
        grid_size = getattr(self.env, "grid_size", 8)
        pos_norm = pos / float(max(1, grid_size - 1))
        res_norm = resources / 100.0
        needs_norm = needs / 100.0
        child_norm = child / 100.0

        flat = np.concatenate([pos_norm, res_norm, needs_norm, child_norm]).astype(np.float32)
        return flat
