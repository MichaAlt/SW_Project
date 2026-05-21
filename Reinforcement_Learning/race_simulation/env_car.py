import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from car import Car
from map_loader import load_map


class CarEnv(gym.Env):
    def __init__(self, map_path, screen_w=1980, screen_h=1080):
        super().__init__()

        pygame.init()

        self.screen, self.game_map, self.display_map, self.scale, self.offset_x, self.offset_y = load_map(
            map_path, screen_w, screen_h
        )

        self.car = Car()

        # OBS = 5 Sensorwerte
        self.observation_space = spaces.Box(
            low=0,
            high=300,
            shape=(5,),
            dtype=np.float32
        )

        # ACTION = [steering, speed]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.clock = pygame.time.Clock()

    def _get_obs(self):
        return np.array(self.car.radar_values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset()
        self.car.update(self.game_map)
        return self._get_obs(), {}

    def step(self, action):
        steer = float(action[0])
        speed = float(action[1])

        # Skalierung
        self.car.angle += steer * 5
        self.car.speed = speed * 5

        self.car.update(self.game_map)

        obs = self._get_obs()

        reward = 0.0

        # -------------------------
        # TOD BESTRAFEN
        # -------------------------
        if not self.car.alive:
            reward -= 100

        # -------------------------
        # VORWÄRTSBEWEGUNG BELONEN
        # -------------------------
        reward += self.car.speed * 0.05

        # -------------------------
        # SENSORWERTE BELONEN
        # Große Distanz zur Wand = gut
        # -------------------------
        reward += sum(self.car.radar_values) * 0.002

        # -------------------------
        # LENKEN BESTRAFEN
        # Zu starkes Zappeln verhindern
        # -------------------------
        reward -= abs(steer) * 0.01

        # -------------------------
        # SEHR LANGSAMES FAHREN BESTRAFEN
        # Damit er nicht stehen bleibt
        # -------------------------
        if self.car.speed < 0.2:
            reward -= 0.05

        # -------------------------
        # OPTIONAL:
        # Rückwärtsfahren bestrafen
        # -------------------------
        if self.car.speed < 0:
            reward -= 0.5


        terminated = not self.car.alive
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))
        self.car.draw(self.screen, pygame.font.SysFont("Arial", 24), self.scale, self.offset_x, self.offset_y)

        pygame.display.flip()
        self.clock.tick(60)