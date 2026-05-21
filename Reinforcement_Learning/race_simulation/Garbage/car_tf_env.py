import numpy as np
import pygame

from car import Car
from map_loader import load_map


BORDER_COLOR = (255, 255, 255)


class CarTFEnv:
    def __init__(self, map_path, width=1280, height=720):
        pygame.init()

        self.screen, self.game_map, self.display_map, self.scale, self.offset_x, self.offset_y = load_map(
            map_path, width, height
        )

        self.car = Car()

        # Spaces (TF-Agents braucht explizit shapes)
        self.obs_shape = (5,)
        self.action_shape = (2,)

        self.reset()

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _get_obs(self):
        return np.array(self.car.radar_values, dtype=np.float32)

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.car.reset()
        self.done = False
        self.steps = 0
        return self._get_obs()

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        self.steps += 1

        speed = float(action[0])
        steer = float(action[1])

        self.car.speed = speed
        self.car.angle += steer
        self.car.update(self.game_map)

        obs = self._get_obs()

        # -------------------------
        # REWARD FUNCTION
        # -------------------------
        reward = 0.0

        # forward reward
        reward += speed * 0.1

        # survival reward
        if len(self.car.radar_values) > 0:
            reward += min(self.car.radar_values) / 1000.0

        # crash penalty
        if not self.car.alive:
            reward -= 100.0
            self.done = True

        # time penalty
        reward -= 0.01

        if self.steps > 2000:
            self.done = True

        return obs, reward, self.done