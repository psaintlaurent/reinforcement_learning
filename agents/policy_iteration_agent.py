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
        self.policy = policy.MCFirstVisitPolicy()
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
    parser.add_argument('env_id', nargs='?', default='Taxi-v3', help='Select the environment to run.')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)


    """ 
        To deal with openai gym I'm setting the standard that an env must be reset at the beginning of a method and cleaned up at the end of a method.
        If you pass an env object it should be safe to reset it and start working with it after a method call.
    """
    agent = PolicyIterationAgent()
    policy = policy.MCFirstVisitPolicy()
    agent.seed(policy, env)
    action = env.action_space.sample()

    current_state = env.reset()
    observation, reward, done, _ = env.step(action)

    print("Start test of the policy")
    action = env.action_space.sample()

    g = 0

    for cnt in range(80000):
        if action is None:
            print("No state-action available in the policy")
            continue



        observation, reward, done, _ = env.step(action)

        print(observation)
        current_state = observation
        env.render()

        print("Policy test, State: %s Action: %s Reward: %s" % (current_state, action, reward))
        prev_action, prev_reward = action, reward




        if done:
            print("Policy test done after exceeding boundary.")
            env.reset()
            break

        action, reward = policy.next_action(current_state)

    env.close()
    exit()
