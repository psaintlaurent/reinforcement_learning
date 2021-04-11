from collections import OrderedDict
import math
from numpy.random import default_rng
import random

"""
The general idea is that this will eventually be a generic policy object that is managed by a Policy Iteration Agent which passes in the specific control and prediction 
#TODO implement object serialization/deserialization so that.
"""


class Policy(object):
    pass


class MCFirstVisitPolicy(Policy):
    observations = None
    optimal_state_action = None
    time_steps = None
    gamma = None  # discount rate
    epsilon = None
    precision = None
    rng = None

    def __init__(self):
        self.observations = OrderedDict()
        self.optimal_state_action = OrderedDict()
        self.time_steps = 100000
        self.gamma = .2
        self.epsilon = .3
        self.precision = 2
        self.rng = default_rng(seed=2192158217327913210321738232)

    """
           Initialize the policy by generating an episode following pi, an initially random epsilon soft policy. The code for this is going to look odd.
    """

    def initialize(self, env):

        observation = env.reset()
        current_state = observation
        action = None
        count = self.time_steps

        while count > 0:

            count -= 1
            random_number = self.rng.random()

            if random_number >= self.epsilon and current_state is not None:   action, _ = self.next_action(current_state)
            if action is None or random_number < self.epsilon:    action, _ = self.next_action_random(env, current_state)

            observation, reward, done, _ = env.step(action)

            if done:    observation = env.reset()
            if current_state is not None:   self.add(current_state, action, reward)

            if count % 5000 == 0:
                pass
            current_state, previous_reward, previous_action = observation, reward, action

        """ cleanup environment """
        while not done:
            observation, reward, done, _ = env.step(action)

        return

    def next_action_random(self, env, current_state):
        actions_taken = []
        if current_state in self.observations and len(self.observations[current_state]) > 0:
            actions_taken = [self.observations[current_state][idx]['next_action'] for idx in range(len(self.observations[current_state]))]

        action = env.action_space.sample()

        if len(actions_taken) < env.action_space.n:
            while action in actions_taken:
                action = env.action_space.sample()

        return action, None

    """
        This is a slightly better version but still inefficient method of keeping track of mappings of observations to actions but I'll worry about that later.
    """

    def add(self, observation, next_action, reward):

        if observation not in self.observations:
            self.observations[observation] = []

        idx = -1
        for iidx in range(len(self.observations[observation])):
            if self.observations[observation][iidx]['next_action'] == next_action:
                idx = iidx

        if idx == -1:
            self.observations[observation].append({
                'next_action'       : next_action,
                'next_reward'       : reward,
                'discounted_returns': [0],
                'visits'            : 1,
                'probability'       : .000000000
            })

            self.observations[observation][idx]['average_returns'] = sum(self.observations[observation][idx]['discounted_returns']) / len(self.observations[observation][idx]['discounted_returns'])

        return

    def set_optimal_state_action(self, observation):

        max_return = 0

        for idx in range(len(self.observations[observation])):

            self.observations[observation][idx]['average_returns'] = sum(self.observations[observation][idx]['discounted_returns']) / len(self.observations[observation][idx]['discounted_returns'])
            if self.observations[observation][idx]['average_returns'] >= max_return:
                max_return = self.observations[observation][idx]['average_returns']

                self.optimal_state_action[observation] = {
                    'action': self.observations[observation][idx]['next_action'],
                    'return': self.observations[observation][idx]['average_returns']
                }

        return

    def next_action(self, observation):

        best_next_action, best_next_reward = None, None

        random_number = self.rng.random()

        """ These may need to be updated at some point """
        greedy_action = 1 - self.epsilon
        non_greedy_action = self.epsilon

        if greedy_action > random_number:

            if observation in self.observations:
                for observation_data in self.observations[observation]:

                    next_action, next_reward = observation_data['next_action'], observation_data['average_returns']
                    if best_next_action is None or next_reward > best_next_reward:
                        best_next_action, best_next_reward = next_action, next_reward
                        self.set_optimal_state_action(observation)

        elif non_greedy_action <= random_number:

            if observation in self.observations:
                keys = [i for i in range(len(self.observations[observation]))]
                self.rng.shuffle(keys)
                best_next_action, best_next_reward = self.observations[observation][keys[0]]['next_action'], self.observations[observation][keys[0]]['average_returns']

        return best_next_action, best_next_reward

    def update_optimal_state_actions(self, observation, action_idx):

        self.observations[observation][action_idx]['average_returns'] = sum(self.observations[observation][action_idx]['discounted_returns']) / \
                                                                        len(self.observations[observation][action_idx]['discounted_returns'])

        if observation in self.optimal_state_action:
            if self.observations[observation][action_idx]['average_returns'] >= self.optimal_state_action[observation]['return']:
                self.optimal_state_action[observation]['action'] = self.observations[observation][action_idx]['next_action']
                self.optimal_state_action[observation]['return'] = self.observations[observation][action_idx]['average_returns']
        else:
            self.set_optimal_state_action(observation)

        return

    def get_action_idx(self, observation, action):

        idx = -1
        if observation in self.observations:
            for iidx in range(len(self.observations[observation])):
                if self.observations[observation][iidx]['next_action'] == action:
                    idx = iidx

        return idx

    def inc_visit(self, state, action):

        idx = self.get_action_idx(state, action)
        if state in self.observations and self.observations[state][idx]['next_action'] == action:
            self.observations[state][idx]['visits'] += 1

        return

    """ Generate episode according to the policy """

    def generate_episode(self, env, max_length=100):

        cnt, done, episode = 0, False, []
        rng = default_rng(215219)

        random_number = rng.random()

        greedy_action = 1 - self.epsilon
        non_greedy_action = self.epsilon
        observation = env.reset()

        while cnt < max_length and not done:

            if greedy_action > random_number:
                action, _ = self.next_action(observation)
            if non_greedy_action <= random_number:
                action, _ = self.next_action_random(env, observation)
            cnt += 1

            state = observation
            observation, reward, done, _ = env.step(action)
            if non_greedy_action <= random_number:
                self.add(state, action, reward)

            self.inc_visit(state, action)
            episode.append((state, action, reward))

        """ cleanup environment """
        while not done:
            observation, reward, done, _ = env.step(action)

        return episode

    def eval(self, env):

        self.initialize(env)

        g = 0
        for i in range(self.time_steps):

            episode = self.generate_episode(env)
            appeared_in_episode = OrderedDict()

            if i % 100000 == 0:
                pass

            for idx in range(len(episode) - 1):

                state, action, reward = episode[idx]
                next_state, next_action, next_reward = episode[idx + 1]

                """ Add any new observations. """
                if len(self.observations[state]) == 0:
                    self.add(state, action, reward)

                action_idx = self.get_action_idx(state, action)

                """ This may have a bug in the event a new observation appears in the environment, this is just a start. """

                g = (g * self.gamma) + next_reward
                if (state, action) not in appeared_in_episode:
                    self.observations[state][action_idx]['discounted_returns'].append(g)
                    appeared_in_episode[(state, action)] = True


                self.set_optimal_state_action(state)
                self.update_optimal_state_actions(state, action_idx)

        return
