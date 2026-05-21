from stable_baselines3 import PPO
from env_car import CarEnv

MAP_PATH = "PNG_File/map1.png"

env = CarEnv(MAP_PATH)

model = PPO.load("car_rl_model")

obs, _ = env.reset()

running = True
while running:
    action, _states = model.predict(obs)

    obs, reward, terminated, truncated, _ = env.step(action)

    env.render()

    if terminated or truncated:
        obs, _ = env.reset()