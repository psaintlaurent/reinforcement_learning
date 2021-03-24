from collections import OrderedDict

"""
This needs major refactoring
The general idea is that this will eventually be a generic policy object
that is managed by a Policy Iteration Agent
"""


class Policy(object):
    observations = None
    returns = None
    returns_count = None
    optimal_state_action = None

    def __init__(self):
        self.observations = OrderedDict()
        self.returns = OrderedDict()
        self.returns_count = OrderedDict()
        self.optimal_state_action = OrderedDict()

    """
        This is a slightly better version but still inefficient method of keeping track of mappings of
        observations to actions but I'll worry about that later.
    """

    def add(self, observation, reward, next_action):

        observation = tuple(observation)
        if observation not in self.observations:
            self.observations[observation] = [{
                'next_action': next_action,
                'next_reward': reward,
                'discounted_returns': [],
                'average_returns': sum(self.observations[observation]) / len(self.observations[observation]),
                'visits': None,
                'probability': None
            }]

        else:
            self.observations[observation].append({
                'next_action': next_action,
                'next_reward': reward,
                'discounted_returns': [],
                'average_returns': sum(self.observations[observation]) / len(self.observations[observation]),
                'visits': None,
                'probability': None
            })

        return

    def set_optimal_state_action(self, observation, action):

        observation = tuple(observation)
        if observation not in self.optimal_state_action or self.optimal_state_action[observation]['return'] <= \
                self.returns[(observation, action)]:
            self.optimal_state_action[observation] = {'action': action, 'return': self.returns[(observation, action)]}

        return

    def next_action(self, observation):

        observation = tuple(observation)
        best_next_action, best_next_reward = None, None

        if observation in self.optimal_state_action:
            best_next_action = self.optimal_state_action[observation]['action']
            best_next_reward = self.optimal_state_action[observation]['return']

        if observation in self.observations:
            for next_action, next_reward, _, _, _ in self.observations[observation]:
                if best_next_action is None or next_reward > best_next_reward:
                    best_next_action, best_next_reward = next_action, next_reward

        return best_next_action, best_next_reward
