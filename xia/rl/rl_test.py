from stable_baselines3.common.env_checker import check_env
from SW_Project.xia.rl.rl_env import CarEnv


env = CarEnv()

print("开始检查环境...")

check_env(env, warn=True)

print("环境检查通过")

obs, info = env.reset()
print("Start observation:", obs)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("Step:", i)
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    if terminated or truncated:
        obs, info = env.reset()