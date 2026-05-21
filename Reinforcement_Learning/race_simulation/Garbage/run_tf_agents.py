import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment

from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network, value_network

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from tf_agents.utils import common

from SW_Project.Reinforcement_Learning.race_simulation.Garbage.car_tf_env import CarTFEnv


MAP_PATH = "../race_simulation/PNG_File/map5.png"


# -------------------------
# WRAP ENV INTO TF-AGENTS ENV
# -------------------------
class PyCarEnvWrapper(py_environment.PyEnvironment):
    def __init__(self):
        self.env = CarTFEnv(MAP_PATH)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,),
            dtype=np.float32,
            minimum=0.0,
            maximum=300.0,
            name="observation"
        )

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action"
        )

        self._state = self.env.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.env.reset()
        return ts.restart(self._state)

    def _step(self, action):
        obs, reward, done = self.env.step(action)

        self._state = obs

        if done:
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=0.99)


# -------------------------
# TRAINING
# -------------------------
def main():

    py_env = PyCarEnvWrapper()
    train_env = tf_py_environment.TFPyEnvironment(py_env)

    # Networks
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(256, 128, 64)
    )

    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(256, 128, 64)
    )

    # PPO Agent
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        train_step_counter=tf.Variable(0)
    )

    agent.initialize()

    # Policy
    policy = agent.policy
    collect_policy = agent.collect_policy

    print("Training started (TF-Agents PPO)...")

    # VERY simplified training loop (important!)
    for episode in range(1000):

        time_step = train_env.reset()
        episode_reward = 0

        for _ in range(500):

            action_step = collect_policy.action(time_step)
            time_step = train_env.step(action_step.action)

            episode_reward += time_step.reward.numpy()[0]

            if time_step.is_last():
                break

        print(f"Episode {episode} Reward: {episode_reward}")

    # Save model
    tf.saved_model.save(agent.policy, "car_tf_agents_model")
    print("Model saved!")


if __name__ == "__main__":
    main()