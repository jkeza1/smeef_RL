import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Tuple, List, Optional
from enum import Enum

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    USE_SERVICE = 4
    WORK_PART_TIME = 5
    ATTEND_TRAINING = 6
    SEEK_SUPPORT = 7

class SMEEFEnv(gym.Env):
    def __init__(self, grid_size=8, max_steps=200, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.last_action = None
        
        # Define action space
        self.action_space = spaces.Discrete(len(Action))
        
        # Dict observation space
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'resources': spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
            'needs': spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
            'child_status': spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        })
        
        # Initialize state
        self.position = None
        self.resources = None  # [money, energy, skills, social_support]
        self.needs = None      # [childcare, financial, emotional, career]
        self.child_status = None  # [health, happiness]
        
        # Service locations
        self.services = {
            'CHILDCARE': {'positions': [(1, 1), (1, 2)], 'color': (173, 216, 230)},
            'EDUCATION': {'positions': [(6, 1), (6, 2)], 'color': (144, 238, 144)},
            'FINANCIAL': {'positions': [(1, 6), (2, 6)], 'color': (255, 255, 150)},
            'HEALTHCARE': {'positions': [(6, 6), (5, 6)], 'color': (255, 182, 193)},
            'COMMUNITY': {'positions': [(3, 3)], 'color': (221, 160, 221)},
            'COUNSELING': {'positions': [(4, 4)], 'color': (152, 251, 152)}
        }
        
        self.home_location = (0, 0)
        self.work_location = (7, 0)
        self.goal_location = (7, 7)
        
        # Visualization
        self.window = None
        self.clock = None

    def _get_observation(self):
        return {
            'position': self.position,
            'resources': self.resources,
            'needs': self.needs,
            'child_status': self.child_status
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0, 0], dtype=np.int32)
        self.resources = np.array([30.0, 70.0, 25.0, 20.0], dtype=np.float32)
        self.needs = np.array([60.0, 70.0, 50.0, 80.0], dtype=np.float32)
        self.child_status = np.array([80.0, 60.0], dtype=np.float32)
        self.current_step = 0
        self.last_action = None
        return self._get_observation(), {}

    def _is_valid_action(self, action):
        """Check if action is valid for current position"""
        # Convert action to int if it's a numpy array
        action_int = int(action) if isinstance(action, (np.ndarray, np.integer)) else action
        
        pos_tuple = tuple(self.position)
        
        # Define what actions are available where
        position_constraints = {
            (0, 0): [Action.MOVE_RIGHT.value, Action.MOVE_DOWN.value, Action.USE_SERVICE.value],  # Home
            (7, 0): [Action.MOVE_LEFT.value, Action.MOVE_DOWN.value, Action.WORK_PART_TIME.value],  # Work
            (7, 7): [Action.MOVE_LEFT.value, Action.MOVE_UP.value],  # Goal
        }
        
        # Service locations allow USE_SERVICE
        for service_name, service_info in self.services.items():
            for service_pos in service_info['positions']:
                if service_pos not in position_constraints:
                    position_constraints[service_pos] = []
                position_constraints[service_pos].extend([
                    Action.MOVE_UP.value, Action.MOVE_DOWN.value, 
                    Action.MOVE_LEFT.value, Action.MOVE_RIGHT.value,
                    Action.USE_SERVICE.value
                ])
        
        # If current position has constraints, check them
        if pos_tuple in position_constraints:
            return action_int in position_constraints[pos_tuple]
        
        # Default: all movement actions are valid
        return action_int in [Action.MOVE_UP.value, Action.MOVE_DOWN.value, 
                             Action.MOVE_LEFT.value, Action.MOVE_RIGHT.value]

    def step(self, action):
        # Convert action to int if it's a numpy array
        action_int = int(action) if isinstance(action, (np.ndarray, np.integer)) else action
        # record pre-step state for diagnostics
        old_resources = self.resources.copy()
        old_needs = self.needs.copy()
        old_child = self.child_status.copy()

        # Detect service at current position (for info)
        pos_tuple = tuple(self.position)
        service_at_pos = None
        for s_name, s_info in self.services.items():
            if pos_tuple in s_info['positions']:
                service_at_pos = s_name
                break

        # Action validation
        if not self._is_valid_action(action_int):
            reward = -1.0  # Penalty for invalid action (smaller, to avoid overshadowing learning signal)
            self.current_step += 1
            info = {
                'invalid_action': True,
                'service_used': None,
                'reward_components': {}
            }
            return self._get_observation(), reward, False, False, info

        self.current_step += 1
        reward = 0
        terminated = False
        truncated = False

        # ENERGY COSTS FOR EACH ACTION
        energy_costs = {
            Action.MOVE_UP.value: 3,
            Action.MOVE_DOWN.value: 3,
            Action.MOVE_LEFT.value: 3,
            Action.MOVE_RIGHT.value: 3,
            Action.USE_SERVICE.value: 8,
            Action.WORK_PART_TIME.value: 12,
            Action.ATTEND_TRAINING.value: 8,
            Action.SEEK_SUPPORT.value: 6
        }
        
        # Apply energy cost - USE action_int
        if action_int in energy_costs:
            self.resources[1] -= energy_costs[action_int]  # resources[1] is energy
            self.resources[1] = max(0, min(100, self.resources[1]))  # Cap energy
        
        # Penalty for low energy - using a smaller penalty to encourage recovery behavior
        if self.resources[1] <= 10 and action_int != Action.USE_SERVICE.value:
            reward -= 1.0
        
        # Penalty for repeating same action - reduce slightly to avoid too strong discouragement
        if self.last_action is not None and action_int == self.last_action:
            reward -= 0.2
        self.last_action = action_int

        # Execute action - USE action_int
        if action_int == Action.MOVE_UP.value:
            reward += self._move_agent(0, -1)
        elif action_int == Action.MOVE_DOWN.value:
            reward += self._move_agent(0, 1)
        elif action_int == Action.MOVE_LEFT.value:
            reward += self._move_agent(-1, 0)
        elif action_int == Action.MOVE_RIGHT.value:
            reward += self._move_agent(1, 0)
        elif action_int == Action.USE_SERVICE.value:
            reward += self._use_service()
        elif action_int == Action.WORK_PART_TIME.value:
            reward += self._work_part_time()
        elif action_int == Action.ATTEND_TRAINING.value:
            reward += self._attend_training()
        elif action_int == Action.SEEK_SUPPORT.value:
            reward += self._seek_support()

        # Natural decay
        self.resources[1] = max(0, self.resources[1] - 0.5)  # Energy decay
        self.needs[0] = min(100, self.needs[0] + 0.3)        # Childcare needs increase
        self.needs[1] = min(100, self.needs[1] + 0.2)        # Financial stress increases

        # Check termination
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 10

        if (self.position == self.goal_location).all():
            goal_bonus = self._calculate_goal_bonus()
            reward += goal_bonus
            terminated = True

        # Check critical conditions
        if (self.resources[1] <= 0 or self.child_status[0] <= 0 or self.resources[0] <= -50):
            reward -= 20
            terminated = True
        # Build info dict with component breakdown
        resource_delta = (self.resources - old_resources).astype(float)
        need_reduction = (old_needs - self.needs).astype(float)  # positive if reduced
        child_delta = (self.child_status - old_child).astype(float)

        reward_components = {
            'money_delta': float(resource_delta[0]),
            'energy_delta': float(resource_delta[1]),
            'skills_delta': float(resource_delta[2]),
            'support_delta': float(resource_delta[3]),
            'need_reduction_total': float(np.sum(need_reduction)),
            'child_health_delta': float(child_delta[0]),
            'child_happiness_delta': float(child_delta[1]),
        }

        info = {
            'invalid_action': False,
            'service_used': service_at_pos,
            'reward_components': reward_components
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _move_agent(self, dx, dy):
        new_pos = self.position + np.array([dx, dy])
        if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            self.position = new_pos
            return -0.05  # Small movement cost (normalized)
        return -0.5  # Penalty for hitting wall

    def _use_service(self):
        pos_tuple = tuple(self.position)
        reward = 0
        
        # Check if at any service location
        for service_name, service_info in self.services.items():
            if pos_tuple in service_info['positions']:
                if service_name == 'CHILDCARE':
                    reward += self._update_need('childcare', -25)
                    reward += self._update_resource('energy', -10)
                elif service_name == 'EDUCATION':
                    reward += self._update_resource('skills', 20)
                    reward += self._update_need('career', -15)
                elif service_name == 'FINANCIAL':
                    reward += self._update_resource('money', 40)
                    reward += self._update_need('financial', -25)
                elif service_name == 'HEALTHCARE':
                    reward += self._update_resource('energy', 20)
                    reward += self._update_child_status('health', 15)
                elif service_name == 'COMMUNITY':
                    reward += self._update_resource('support', 25)
                    reward += self._update_need('emotional', -20)
                elif service_name == 'COUNSELING':
                    reward += self._update_need('emotional', -25)
                    reward += self._update_resource('support', 15)
                # Small fixed bonus for successfully using a service (kept small)
                return reward + 1.0
        return -1  # Penalty for using service nowhere

    def _work_part_time(self):
        if (self.position == self.work_location).all():
            reward = self._update_resource('money', 30)
            reward += self._update_resource('energy', -25)
            reward += self._update_need('childcare', 15)  # Increased childcare need
            return reward + 1.0
        return -1  # Penalty for working elsewhere

    def _attend_training(self):
        reward = self._update_resource('skills', 15)
        reward += self._update_resource('energy', -15)
        reward += self._update_need('career', -10)
        return reward

    def _seek_support(self):
        reward = self._update_resource('support', 20)
        reward += self._update_need('emotional', -15)
        return reward

    def _update_resource(self, resource, change):
        # Return reward proportional to the actual change (normalized) with resource-specific weights.
        resource_map = {'money': 0, 'energy': 1, 'skills': 2, 'support': 3}
        weight_map = {'money': 1.0, 'energy': 0.3, 'skills': 0.8, 'support': 0.5}
        if resource in resource_map:
            idx = resource_map[resource]
            old_value = float(self.resources[idx])
            self.resources[idx] = np.clip(self.resources[idx] + change, 0, 100)
            new_value = float(self.resources[idx])
            delta = new_value - old_value
            # Normalize delta to [-1,1] by dividing by 100 and scale by weight
            return (delta / 100.0) * weight_map.get(resource, 0.5)
        return 0.0

    def _update_need(self, need, change):
        # Positive reward when a need is reduced. Reward is proportional to the amount reduced.
        need_map = {'childcare': 0, 'financial': 1, 'emotional': 2, 'career': 3}
        weight_map = {'childcare': 1.0, 'financial': 1.0, 'emotional': 0.8, 'career': 0.6}
        if need in need_map:
            idx = need_map[need]
            old_value = float(self.needs[idx])
            self.needs[idx] = np.clip(self.needs[idx] + change, 0, 100)
            new_value = float(self.needs[idx])
            # Positive delta = need increased (bad). We reward reductions: old - new
            delta_reduction = old_value - new_value
            return (delta_reduction / 100.0) * weight_map.get(need, 0.8)
        return 0.0

    def _update_child_status(self, status, change):
        # Reward proportional to improvements in child's status (health/happiness)
        status_map = {'health': 0, 'happiness': 1}
        weight_map = {'health': 0.8, 'happiness': 1.0}
        if status in status_map:
            idx = status_map[status]
            old_value = float(self.child_status[idx])
            self.child_status[idx] = np.clip(self.child_status[idx] + change, 0, 100)
            new_value = float(self.child_status[idx])
            delta = new_value - old_value
            return (delta / 100.0) * weight_map.get(status, 0.8)
        return 0.0

    def _calculate_goal_bonus(self):
        # Bounded goal bonus (normalized) so final reward is on a comparable scale
        # Weights chosen to reflect relative importance; total max ~10.0
        bonus = 0.0
        bonus += (float(self.resources[0]) / 100.0) * 2.0   # money weight
        bonus += ((100.0 - float(np.max(self.needs))) / 100.0) * 3.0  # needs management weight
        bonus += (float(self.resources[2]) / 100.0) * 2.0   # skills weight
        bonus += (float(self.child_status[1]) / 100.0) * 3.0  # child happiness weight
        return bonus

    def render(self):
        if self.render_mode == "human":
            # Lazy-import pygame to avoid import overhead when not rendering
            import pygame

            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((800, 800))
                pygame.display.set_caption("SMEEF Environment")
                self.clock = pygame.time.Clock()

            self.window.fill((255, 255, 255))
            cell_size = 800 // self.grid_size
            
            # Draw grid and services
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)
            
            # Draw services
            for service_name, service_info in self.services.items():
                for pos in service_info['positions']:
                    rect = pygame.Rect(pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size)
                    pygame.draw.rect(self.window, service_info['color'], rect)
            
            # Draw special locations
            home_rect = pygame.Rect(0, 0, cell_size, cell_size)
            pygame.draw.rect(self.window, (100, 200, 100), home_rect)
            
            work_rect = pygame.Rect(7 * cell_size, 0, cell_size, cell_size)
            pygame.draw.rect(self.window, (100, 150, 255), work_rect)
            
            goal_rect = pygame.Rect(7 * cell_size, 7 * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.window, (255, 215, 0), goal_rect)
            
            # Draw agent
            agent_rect = pygame.Rect(
                self.position[0] * cell_size + cell_size//4,
                self.position[1] * cell_size + cell_size//4,
                cell_size//2, cell_size//2
            )
            pygame.draw.rect(self.window, (255, 0, 0), agent_rect)
            
            pygame.display.flip()
            self.clock.tick(10)

    def close(self):
        if self.window is not None:
            # Lazily import pygame if we created a window
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass