import argparse
import gym
from . import policy
from gym import wrappers, logger


"""
    The goal here is to eventually have a single policy iteration agent
    that allows for passing in different control and prediction algorithms.
    
    For now I'm sticking with on-policy first visit Monte Carlo control for epsilon soft policies
    
    Created based on pseudocode from Sutton and Barto Bandit Algorithm p. 101
"""


class PolicyIterationAgent(object):
    action_values = None  # a mapping of actions to their values
    state_values = None  # a mapping of states to their values
    policy = None  # a mapping of states to optimal actions
    time_steps = None  # time steps per episode
    gamma = None  # discount rate
    epsilon = None

    def __init__(self):
        self.action_values = {}
        self.state_values = {}
        self.policy = policy.Policy()
        self.time_steps = 20
        self.gamma = .2
        self.epsilon = .1

    def act(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run.')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = PolicyIterationAgent(env.action_space)

    env.close()
