import argparse

import gym
from gym import logger, wrappers

from agents.policies import policy

"""
    The goal here is to eventually have a single policy iteration agent that allows for passing in different control and prediction algorithms.
    For now I'm sticking with on-policy first visit Monte Carlo control for epsilon soft policies.
    Created based on pseudocode from Sutton and Barto Bandit Algorithm p. 101
"""


class PolicyIterationAgent(object):

    action_values = None  # a mapping of actions to their values
    state_values = None  # a mapping of states to their values
    policy = None  # a mapping of states to optimal actions
    time_steps = None  # time steps per episode
    gamma = None  # discount rate
    epsilon = None
    precision = None

    def __init__(self):
        self.action_values = {}
        self.state_values = {}
        self.policy = policy.Policy()
        self.time_steps = 20
        self.gamma = .2
        self.epsilon = .1
        self.precision = 2

    def act(self, env):
        pass

    def seed(self, policy, env):

        self.policy = policy
        self.policy.eval(env)

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run.')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = PolicyIterationAgent()
    policy = policy.Policy()
    agent.seed(policy, env)

    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    observation = tuple([round(val, policy.precision) for val in observation]) # TODO Find a way to handle this centrally along with env.step(c) calls

    print("Start test of the policy")
    action = env.action_space.sample()
    for idx in range(300):

        if action is not None:
            observation, reward, done, _ = env.step(action)
            observation = tuple([round(val, policy.precision) for val in observation])  # TODO Find a way to handle this centrally along with env.step(c) calls
            print("Policy test, State: %s Action: %s Reward: %s" % (observation, action, reward))

            if done:
                print("Policy test done after exceeding boundary.")
                env.reset()
                break

        action, reward = policy.next_action(observation)

    env.close()
