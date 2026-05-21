from stable_baselines3 import PPO
from env_car import CarEnv

MAP_PATH = "PNG_File/map1.png"

env = CarEnv(MAP_PATH)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=50_000)

model.save("car_rl_model")
print("Training fertig!")