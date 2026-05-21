import tensorflow as tf
import numpy as np
import pygame

from SW_Project.Reinforcement_Learning.race_simulation.Garbage.car_tf_env import CarTFEnv

MAP_PATH = "../race_simulation/PNG_File/map5.png"


def main():

    env = CarTFEnv(MAP_PATH)

    policy = tf.saved_model.load("car_tf_agents_model")

    obs = env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:

        obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)

        action = policy.action(obs_tensor).action.numpy()[0]

        obs, reward, done = env.step(action)

        env.render()

        if done:
            obs = env.reset()

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()