import argparse
import gym
import sys
from gym import wrappers, logger
from numpy.random import default_rng

"""
    Created based on pseudocode from Sutton and Barto Bandit Algorithm p. 32
"""

class BanditAgent(object):

    action_values = None
    epsilon = .1

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation=None, reward=None, done=None):

        rng = default_rng(219215)
        random_number = rng.random()

        if random_number < self.epsilon or len(self.action_values) == 0:
            output = self.action_space.sample()
        elif random_number >= self.epsilon:
            output = max(self.action_values)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = BanditAgent(env.action_space)

    episode_count, reward, done = 100, 0, False

    for i in range(episode_count):
        observation = env.reset()
        number_of_selections = {}
        agent.action_values = {}
        step = 0

        while True:

            action = agent.act(observation, reward, done)
            observation, reward, done, _ = env.step(action)
            tpl = tuple(observation)
            step += 1

            print("Step %s: %s, %s, %s" % (step, observation, reward, done))

            if action not in agent.action_values:
                agent.action_values[action] = 0
            if action not in number_of_selections:
                number_of_selections[action] = 0

            number_of_selections[action] += 1

            agent.action_values[action] = agent.action_values[action] + ((1 / number_of_selections[action]) *
                                                                         (reward - agent.action_values[action]))

            if done:
                print("\n\n")
                break

    env.close()