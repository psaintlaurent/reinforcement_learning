import argparse
import gym
import sys
from gym import wrappers, logger
from numpy.random import default_rng

"""
    
    The goal here is to eventually have a single policy iteration agent
    that allows for passing in different control and prediction algorithms.
    
    For now I'm sticking with On-policy first visit Monte Carlo control for epsilon soft policies
    
    Created based on pseudocode from Sutton and Barto Bandit Algorithm p. 101
"""


class PolicyIterationAgent(object):
    action_values = None  # a mapping of actions to their values
    state_values = None  # a mapping of states to their values
    policy = None  # a mapping of states to optimal actions
    returns = None
    gamma = None  # discount rate
    time_steps = None  # time steps per episode

    def __init__(self):
        self.action_values = {}
        self.state_values = {}
        self.policy = Policy()
        self.time_steps = 20
        self.gamma = .2

    def act(self, observation=None, reward=None, done=None):

        if len(self.policy) == 0:
            output = self.actions_space.sample()
        else:
            output = self.actions_space

    def eval(self, env):

        """ Initialize the policy by generating an episode following pi, an initially random epsilon soft policy. """
        for idx in range(self.time_steps):

            if len(self.policy.observations) < self.time_steps:

                while not done:

                    current_action = self.act(observation, reward, done)
                    observation, reward, done, _ = env.step()

                    next_action = self.act(observation, reward, done)
                    observation, reward, done, _ = env.step()

                    """TODO """
                    for i in range(self.time_steps):
                        pass

                    current_action = next_action

        return


class Policy(object):
    observations, next_actions = [], []
    epsilon = None

    def __init__(self):
        self.observations = []
        self.next_actions = []
        self.epsilon = .1

    """ 
    This is a wildly inefficient method of keeping track of mappings of
    observations to actions but I'll worry about that later
    """

    def add(self, observation, reward, next_action):
        if observation not in self.observations:
            self.observations.append(observation)
            self.next_actions.append([next_action])
        else:
            idx = 0
            for val in self.observations:
                if val == observation:
                    self.next_actions[idx].append(next_action)
                    break
                idx += 1

        return

    def act(self, observation, action_space):

        """ Epsilon soft policy """
        rng = default_rng(219215)
        random_number = rng.random()

        if random_number < self.epsilon or len(self.action_values) == 0:
            output = action_space.sample()
        elif random_number >= self.epsilon:

            idx = 0
            """ Slow also but this is just to start, refactor later. """
            for ob in self.observations:
                if observation == ob:
                    break
                idx += 1
            output = self.next_actions[idx]

        return self.next_actions[idx]


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

    episode_count, reward, done = 100, 0, False
    for i in range(episode_count):

        observation = env.reset()
        number_of_selections = {}
        agent.action_values = {}
        step = 0

        while True:

            action = agent.act(observation, reward, done)
            agent.eval(env)

            observation, reward, done, _ = env.step(action)
            step += 1

            print("Step %s: %s, %s, %s" % (step, observation, reward, done))

            if action not in agent.action_values:
                agent.action_values[action] = 0

            if action not in agent.state_values:
                agent.state_values[action] = 0

            if action not in number_of_selections:
                number_of_selections[action] = 0

            agent.action_values[action] += reward
            agent.state_values[action] += reward
            number_of_selections[action] += 1

            if done:
                print("\n\n")
                break

    env.close()
