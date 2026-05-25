import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from pathlib import Path

from car import Car
from map_loader import load_map

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
enviroment_cfg = config["enviroment"]

class enviroment(gym.Env):
    def __init__(self, map_path, screen_w=1920, screen_h=1080, render_mode = False):
        
        # Gym-Enviroment initialisieren
        super().__init__()

        self.render_mode = render_mode
        self.screen = None

        self.max_steps = 50000
        self.step_count = 0


        self.screen, self.game_map, self.display_map, self.scale, self.offset_x, self.offset_y = load_map(
            map_path, screen_w, screen_h
        )

        self.car = Car()

        #TODO: Nicht Rendern funktioniert auch ohne if-Abfrage beim training, wieso??
        if self.render_mode:
            pygame.init()

            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
            # self.screen = pygame.display.set_mode((screen_w, screen_h))
            

        # State definieren
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([20, 300, 300, 300, 300, 300]),
            shape=(6,),
            dtype=np.float32
        )

        # action definieren
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    # State abfragen
    def _get_state(self):
        return np.array([self.car.speed] + self.car.radar_values, dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset()
        self.car.update(self.game_map)
        return self._get_state(), {}

    def step(self, action):
        
        self.step_count += 1

        truncated = False
        terminated = False

        steer = float(action[0])
        speed = float(action[1])

        # Skalierung
        self.car.angle += steer * enviroment_cfg["steer_scaling"] 
        self.car.speed = speed * enviroment_cfg["speed_scaling"] 

        self.car.update(self.game_map)

        obs = self._get_state()

        # Belohnungs-Wert
        reward = 0.0

        # Zusammenstoß mit der Wand bestrafen
        if not self.car.alive:
            reward -= 100

        # Hohe Geschwindigkeit belohnen
        reward += self.car.speed * 0.05

        # Niedrige Geschwindigkeit bestrafen
        if self.car.speed < 0.2:
            reward -= 1
        
        # Starkes lenken bestrafen
        reward -= abs(steer) * 0.01

        # Distanz zur Wand belohnen
        reward += sum(self.car.radar_values) * 0.002

        # Rückwärtsfahren bestrafen
        if self.car.speed < 0:
            reward -= 2
        """
        if self.step_count >= self.max_steps:
            truncated = True
        """
        if not self.car.alive:
            terminated = True
        

        return obs, reward, terminated, truncated, {}

    def render(self):
        
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))
        self.car.draw(self.screen, self.font, self.scale, self.offset_x, self.offset_y)

        pygame.display.flip()
        self.clock.tick(60)