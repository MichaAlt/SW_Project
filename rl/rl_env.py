import os
import sys
from pathlib import Path

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# 项目根目录：SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
RACE_DIR = ROOT_DIR / "race_simulation"

# 让 rl_env.py 可以 import race_simulation 和 Config
sys.path.append(str(ROOT_DIR))
sys.path.append(str(RACE_DIR))

from Config.config_loader import load_config
from car import Car
from map_loader import load_map

def clamp(value, low, high):
    return max(low, min(high, value))

def get_screen_size(cfg):
    pygame.init()
    info = pygame.display.Info()

    if cfg["width"] == "auto" or cfg["height"] == "auto":
        return info.current_w, info.current_h

    return cfg["width"], cfg["height"]

class CarEnv(gym.Env):
    """
    强化学习环境：
    observation = 5 个传感器值
    action = [turn_norm, speed_norm]

    turn_norm: -1 到 1
    speed_norm: 0 到 1
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None, random_map=True, fixed_map=None, sequential_map=False):
        super().__init__()

        self.render_mode = render_mode

        self.config = load_config()
        self.rl_cfg = self.config["rl"]
        self.random_map = random_map
        self.fixed_map = fixed_map

        # 新增：是否按顺序切换地图
        self.sequential_map = sequential_map
        self.map_index = 0

        self.width, self.height = get_screen_size(self.rl_cfg)

        self.map_files = self.rl_cfg["map_files"]
        self.current_map = None

        self.screen = None
        self.game_map = None
        self.display_map = None
        self.scale = None
        self.offset_x = None
        self.offset_y = None

        self.load_map_for_episode()

        pygame.display.set_caption("RL Car Environment")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 24)

        # 5 个传感器输入，归一化后 0~1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        # 连续动作：[turn_norm, speed_norm]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.forward_speed_max = self.rl_cfg["forward_speed_max"]
        self.speed_multiplier = self.rl_cfg["speed_multiplier"]
        self.max_turn = self.rl_cfg["max_turn_per_step"]
        self.sensor_max_distance = self.rl_cfg["sensor_max_distance"]

        self.car = None
        self.score = 0.0
        self.step_count = 0
        self.max_steps = self.rl_cfg["max_steps"]

        self.last_turn_norm = 0.0
        self.last_speed_norm = 0.0
        self.last_turn_angle = 0.0
        self.last_actual_speed = 0.0

        # 新增：跑圈判断相关变量
        self.start_x = 0.0
        self.start_y = 0.0
        self.has_left_start_area = False
        self.lap_finished = False

    def load_map_for_episode(self):
        if self.sequential_map:
            map_file = self.map_files[self.map_index]
            self.map_index = (self.map_index + 1) % len(self.map_files)

        elif self.random_map:
            map_file = random.choice(self.map_files)

        else:
            map_file = self.fixed_map or self.rl_cfg["map_file_run"]

        if not os.path.isabs(map_file):
            map_path = str(RACE_DIR / map_file)
        else:
            map_path = map_file

        self.screen, self.game_map, self.display_map, self.scale, self.offset_x, self.offset_y = load_map(
            map_path,
            self.width,
            self.height
        )
        self.current_map = map_file

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_map_for_episode()
        self.car = Car()
        self.car.reset()
        self.last_x = self.car.center[0]
        self.last_y = self.car.center[1]

        # 新增：记录起点，用来判断是否跑完一圈
        self.start_x = self.car.center[0]
        self.start_y = self.car.center[1]
        self.has_left_start_area = False
        self.lap_finished = False

        self.score = 0.0
        self.step_count = 0

        self.last_turn_norm = 0.0
        self.last_speed_norm = 0.0
        self.last_turn_angle = 0.0
        self.last_actual_speed = 0.0

        # 先 update 一次，让 radar_values 生成出来
        self.car.update(self.game_map)

        observation = self.get_observation()
        info = {
            "score": self.score,
            "current_map": self.current_map,
            "lap_finished": self.lap_finished
        }

        return observation, info

    def step(self, action):
        self.step_count += 1

        # 1. 执行动作
        self.apply_action(action)

        # 2. 读取传感器
        observation = self.get_observation()

        # 3. 判断是否撞墙
        crashed = self.check_collision()

        # 4. 判断是否顺利跑完一圈
        lap_finished = self.check_lap_finished()

        # 5. 撞墙或者跑完一圈，结束当前 episode
        terminated = crashed or lap_finished

        # 6. 不使用 max_steps 截断
        truncated = False

        # 7. 计算 reward
        # 注意：这里只传 crashed，不传 terminated
        # 否则跑完一圈也会被当成撞墙扣分
        reward = self.calculate_reward(observation, crashed)

        self.score += reward

        info = {
            "score": self.score,
            "step_count": self.step_count,
            "current_map": self.current_map,
            "turn_norm": self.last_turn_norm,
            "speed_norm": self.last_speed_norm,
            "turn_angle": self.last_turn_angle,
            "actual_speed": self.last_actual_speed,
            "alive": self.car.alive,
            "crashed": crashed,
            "lap_finished": lap_finished
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        """
        返回真实 5 个 sensor：
        car.radar_values = [right, right_front, front, left_front, left]
        然后除以 sensor_max_distance 做归一化
        """

        if self.car is None or len(self.car.radar_values) != 5:
            return np.array([1, 1, 1, 1, 1], dtype=np.float32)

        observation = np.array(self.car.radar_values, dtype=np.float32)

        observation = observation / self.sensor_max_distance
        observation = np.clip(observation, 0.0, 1.0)

        return observation.astype(np.float32)

    def apply_action(self, action):
        """
        action[0] = turn_norm, 范围 -1 到 1
        action[1] = speed_norm, 范围 0 到 1
        """

        turn_norm = float(action[0])
        speed_norm = float(action[1])

        turn_norm = clamp(turn_norm, -1.0, 1.0)
        speed_norm = clamp(speed_norm, 0.0, 1.0)

        # 和 ai_run.py 一样：turn_norm 还原成角度
        turn_angle = turn_norm * 180.0

        # 和 ai_run.py 一样：speed_norm 还原成真实速度
        actual_speed = (
            speed_norm
            * self.forward_speed_max
            * self.speed_multiplier
        )

        self.car.speed = actual_speed

        # 限制每一步最大转向
        if turn_angle > self.max_turn:
            self.car.angle += self.max_turn
        elif turn_angle < -self.max_turn:
            self.car.angle -= self.max_turn
        else:
            self.car.angle += turn_angle

        # 更新小车位置、碰撞、sensor
        self.car.update(self.game_map)

        self.last_turn_norm = turn_norm
        self.last_speed_norm = speed_norm
        self.last_turn_angle = turn_angle
        self.last_actual_speed = actual_speed

    def check_lap_finished(self):
        """
        判断是否顺利跑完一圈：
        1. 一开始在起点附近，不算完成
        2. 必须先离开起点一定距离
        3. 离开后再次回到起点附近，才算完成一圈
        """

        if self.car is None:
            return False

        current_x = self.car.center[0]
        current_y = self.car.center[1]

        dx = current_x - self.start_x
        dy = current_y - self.start_y
        distance_to_start = (dx ** 2 + dy ** 2) ** 0.5

        leave_start_distance = self.rl_cfg.get("leave_start_distance", 200)
        finish_distance = self.rl_cfg.get("finish_distance", 80)
        min_lap_steps = self.rl_cfg.get("min_lap_steps", 300)

        if distance_to_start > leave_start_distance:
            self.has_left_start_area = True

        if (
            self.has_left_start_area
            and distance_to_start < finish_distance
            and self.step_count > min_lap_steps
        ):
            self.lap_finished = True
            return True

        return False

    def calculate_reward(self, observation, terminated):
        """
    改进版 reward:
    - 撞墙大扣分
    - 活着加分
    - 前方距离远时鼓励加速
    - 前方距离近时如果还很快就扣分
    - 前方距离远但速度太低也扣分
    - 鼓励不要贴墙
        """

        if terminated:
            return float(self.rl_cfg["penalty_crash"])

        right = float(observation[0])
        right_front = float(observation[1])
        front = float(observation[2])
        left_front = float(observation[3])
        left = float(observation[4])

        speed = float(self.last_speed_norm)
        turn = abs(float(self.last_turn_norm))

        reward = 0.0

    # 1. 活着就加分
        reward += self.rl_cfg["reward_alive"]

    # 2. 前方越空，速度越快越奖励
        reward += 2.0 * speed * front
    # 3. 前方很安全，而且不乱转，额外奖励稳定前进

        if front > 0.65 and speed > 0.4 and turn < 0.4:

            reward += 0.5

    # 3. 前方很近还开很快，扣分
        if front < 0.25 and speed > 0.5:
            reward -= 4.0

    # 4. 前方很远但速度太低，扣分
        if front > 0.6 and speed < 0.3:
            reward -= 1.0

    # 6. 鼓励车不要太贴墙
        min_side_distance = min(right, left, right_front, left_front)
        reward += 0.5 * min_side_distance

    # 7. 如果左右距离差太大，说明太靠边，稍微扣分
        side_balance = abs(left - right)
        reward -= 0.4 * side_balance

    # 8. 转向太大且速度很快，容易撞，扣一点
        if abs(turn) > 0.7 and speed > 0.3:
            reward -= 1.0

    # 9. 防止完全不动
        if speed < self.rl_cfg["low_speed_threshold"]:
            reward -= self.rl_cfg["penalty_low_speed"]
    # 10
        current_x = self.car.center[0]
        current_y = self.car.center[1]

        dx = current_x - self.last_x
        dy = current_y - self.last_y
        distance_moved = (dx ** 2 + dy ** 2) ** 0.5

        if distance_moved < 0.5:
            reward -= 0.5

        self.last_x = current_x
        self.last_y = current_y
    # 12. 惩罚左右来回抽动

        reward -= 0.3 * turn

        return float(reward)

    def check_collision(self):
        """
        car.update(game_map) 里如果撞墙，会设置 self.car.alive = False
        """
        return not self.car.alive

    def render(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.display_map, (self.offset_x, self.offset_y))

        if self.car is not None:
            self.car.draw(
                self.screen,
                self.font_small,
                self.scale,
                self.offset_x,
                self.offset_y
            )

    # 显示 RL 当前输出
        font_big = pygame.font.SysFont("Arial", 28)

        map_text = f"Map: {self.current_map}"
        map_surface = font_big.render(map_text, True, (255, 255, 0))
        self.screen.blit(map_surface, (50, 50))

        turn_norm_text = f"turn_norm: {self.last_turn_norm:.3f}"
        turn_norm_surface = font_big.render(turn_norm_text, True, (0, 255, 255))
        self.screen.blit(turn_norm_surface, (50, 90))

        turn_angle_text = f"turn_angle: {self.last_turn_angle:.3f}"
        turn_angle_surface = font_big.render(turn_angle_text, True, (0, 255, 255))
        self.screen.blit(turn_angle_surface, (50, 130))

        speed_norm_text = f"speed_norm: {self.last_speed_norm:.3f}"
        speed_norm_surface = font_big.render(speed_norm_text, True, (0, 255, 255))
        self.screen.blit(speed_norm_surface, (50, 170))

        actual_speed_text = f"actual_speed: {self.last_actual_speed:.3f}"
        actual_speed_surface = font_big.render(actual_speed_text, True, (0, 255, 255))
        self.screen.blit(actual_speed_surface, (50, 210))

        if self.car is not None:
            car_angle_text = f"car.angle: {self.car.angle:.2f}"
            car_angle_surface = font_big.render(car_angle_text, True, (0, 255, 255))
            self.screen.blit(car_angle_surface, (50, 250))

            sensor_text = "Sensor: " + ", ".join(str(v) for v in self.car.radar_values)
            sensor_surface = font_big.render(sensor_text, True, (255, 255, 0))
            self.screen.blit(sensor_surface, (50, 290))

            lap_text = f"lap_finished: {self.lap_finished}"
            lap_surface = font_big.render(lap_text, True, (255, 255, 0))
            self.screen.blit(lap_surface, (50, 330))

        pygame.display.flip()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        pygame.quit()