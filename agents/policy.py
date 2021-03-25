from collections import OrderedDict
from numpy.random import default_rng

"""
The general idea is that this will eventually be a generic policy object that is managed by a Policy Iteration Agent which passes in the specific control and prediction 
"""


class Policy(object):
    observations = None
    optimal_state_action = None
    time_steps = None
    gamma = None  # discount rate
    epsilon = None

    def __init__(self):
        self.observations = OrderedDict()
        self.optimal_state_action = OrderedDict()
        self.time_steps = 20
        self.gamma = .2
        self.epsilon = .1

    """
           Initialize the policy by generating an episode following pi, an initially random epsilon soft policy. The code for this is going to look odd.
    """
    def initialize(self, env):

        """
            Select the epsilon greedy action according to the policy. If no observation was found choose the next action at random.
        """
        rng = default_rng(219215)
        random_number = rng.random()

        observation = env.reset()
        current_state = observation
        action = None

        for idx in range(self.time_steps):

            """ Select an action """
            if random_number < self.epsilon:  # Explore
                action = env.action_space.sample()
            elif random_number >= self.epsilon:  # Exploit if there is an exploitable option
                action, _ = self.next_action(current_state)
                if action is None:
                    action = env.action_space.sample()

            """ Take the selected action """
            observation, reward, done, _ = env.step(action)

            """ Add a new state -> action mapping to the existing policy """
            if current_state is not None:
                self.add(current_state, reward, action)

            """ Store the existing state, reward and action """
            current_state, previous_reward, previous_action = observation, reward, action

        return

    """
        This is a slightly better version but still inefficient method of keeping track of mappings of observations to actions but I'll worry about that later.
    """

    def add(self, observation, reward, next_action):

        observation = tuple(observation)
        if observation not in self.observations:
            self.observations[observation] = [{
                'next_action'    : next_action, 'next_reward': reward, 'discounted_returns': [],
                'average_returns': sum(self.observations[observation]['discounted_returns']) / len(self.observations[observation]['discounted_returns']),
                'visits'         : None, 'probability': None
            }]

        else:
            self.observations[observation].append({
                'next_action'    : next_action, 'next_reward': reward, 'discounted_returns': [],
                'average_returns': sum(self.observations[observation]['discounted_returns']) / len(self.observations[observation]['discounted_returns']),
                'visits'         : None, 'probability': None
            })

        return

    def set_optimal_state_action(self, observation, action):

        observation = tuple(observation)

        for idx in range(len(observation)):
            if observation[idx]['next_action'] == action:
                max_return = max(observation[idx]['discounted_returns'])
        """
            Set the new optimal state action
        """
        if observation not in self.optimal_state_action or self.optimal_state_action[observation]['return'] < max_return:
            self.optimal_state_action[observation] = {
                'action': action,
                'return': max_return
            }

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

    def eval(self, env):

        observation = env.reset()
        self.initialize(env)

        """
            The number of time steps and the number of observations should be equal after initialization
        """
        count = self.time_steps - 1
        ob_iter = reversed(self.observations.keys())

        """
            TODO: extrapolate to N-step
        """
        while count > 0:

            next_ob = self.observations[next(ob_iter)]
            current_ob = self.observations[next(ob_iter)]

            current_next_action, current_reward = self.next_action(current_ob)
            next_next_action, next_reward = self.next_action(next_ob)

            """
                Calculate the first visit return for each (S_T, A_T) when (S_T, A_T) does not appear in the policy
            """
            for idx in self.observations[current_ob]:

                """ Skip iteration in the event there is existing state-action. """
                if self.observations[current_ob][idx]['next_action'] == current_next_action:
                    continue

                """ Calculate the first visit discounted return for the state-action and increment the number of visits. """
                if len(self.observations[current_ob][idx]['discounted_returns']) == self.observations[current_ob][idx]['visits']:
                    self.observations[current_ob][idx]['discounted_returns'].append(self.gamma * 0 + next_reward)
                    self.observations[current_ob][idx]['visits'] += 1
                else:
                    self.observations[current_ob][idx]['discounted_returns'][-1] = self.gamma * self.observations[current_ob][idx]['discounted_returns'] + next_reward

            count -= 1

            self.set_optimal_state_action(current_ob, current_next_action)

        return
