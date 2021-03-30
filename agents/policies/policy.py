from collections import OrderedDict

from numpy.random import default_rng

"""
The general idea is that this will eventually be a generic policy object that is managed by a Policy Iteration Agent which passes in the specific control and prediction 
#TODO implement object serialization/deserialization so that 
"""


class Policy(object):
    observations = None
    optimal_state_action = None
    time_steps = None
    gamma = None  # discount rate
    epsilon = None
    precision = None

    def __init__(self):
        self.observations = OrderedDict()
        self.optimal_state_action = OrderedDict()
        self.time_steps = 4000000
        self.gamma = .2
        self.epsilon = .1
        self.precision = 2


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
        observation = tuple([round(val, self.precision) for val in observation])
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
            observation = tuple([round(val, self.precision) for val in observation])

            """ Add a new state -> action mapping to the existing policy """
            if current_state is not None:
                self.add(current_state, reward, action)

            """ Store the existing state, reward and action """
            current_state, previous_reward, previous_action = observation, reward, action

            if done:
                env.reset()

        return

    """
        This is a slightly better version but still inefficient method of keeping track of mappings of observations to actions but I'll worry about that later.
    """
    def add(self, observation, reward, next_action):

        if observation not in self.observations:
            self.observations[observation] = []

        idx = -1
        for iidx in range(len(self.observations[observation])):
            if self.observations[observation][idx]['next_action'] == next_action:
                idx = iidx

        if idx == -1:
            self.observations[observation].append({
                'next_action'    : next_action, 'next_reward': reward, 'discounted_returns': [],
                'visits'         : 0, 'probability': .000000000
            })

            if len(self.observations[observation][idx]['discounted_returns']) == 0:
                self.observations[observation][idx]['average_returns'] = 0
            else:
                self.observations[observation][idx]['average_returns'] = sum(self.observations[observation][idx]['discounted_returns']) / len(self.observations[observation][idx]['discounted_returns'])

        return

    def set_optimal_state_action(self, observation, action):

        observation = tuple([round(val, self.precision) for val in observation])
        max_return = 0

        for idx in range(len(self.observations[observation])):
            if self.observations[observation][idx]['next_action'] == action:
                if len(self.observations[observation][idx]['discounted_returns']) > 0:
                    max_return = max(self.observations[observation][idx]['discounted_returns'])

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
            for observation_data in self.observations[observation]:
                next_action, next_reward = observation_data['next_action'], observation_data['next_reward']
                if best_next_action is None or next_reward > best_next_reward:
                    best_next_action, best_next_reward = next_action, next_reward

        return best_next_action, best_next_reward

    def eval(self, env):

        observation = env.reset()
        observation = tuple([round(val, self.precision) for val in observation]) # TODO Find a way to handle this centrally along with env.step(c) calls
        self.initialize(env)

        """
            The number of time steps and the number of observations should be equal after initialization
        """
        count = len(self.observations)

        """
            Setup two iterators so that I can simulate an overlapping offset sliding window where [0][1] [1][2] [T][T+1] 
            is captured on each iteration of the while loop.
        """
        ob_iter_1 = reversed(self.observations.keys())
        ob_iter_2 = reversed(self.observations.keys())
        next(ob_iter_1) #offset by one

        """
            TODO: extrapolate to N-step
        """
        while count > 2:

            next_ob = next(ob_iter_1)
            next_ob_data = self.observations[next_ob]

            current_ob = next(ob_iter_2)
            current_ob_data = self.observations[current_ob]

            print("Time Step Countdown: %s, ob: %s ob+1: %s: \n" % (count, next_ob, current_ob))

            current_next_action, current_reward = self.next_action(current_ob)
            next_next_action, next_reward = self.next_action(next_ob)

            """
                Calculate the first visit return for each (S_T, A_T) when (S_T, A_T) does not appear in the policy
            """
            for idx in range(len(self.observations[current_ob])):

                """ 
                    Skip iteration in the event there is existing state-action.
                """
                if self.observations[current_ob][idx]['next_action'] == current_next_action:
                    continue

                """
                    Calculate the first visit discounted return for the state-action and increment the number of visits. 
                """
                if len(self.observations[current_ob][idx]['discounted_returns']) == self.observations[current_ob][idx]['visits']:
                    self.observations[current_ob][idx]['discounted_returns'].append(self.gamma * 0 + next_reward)
                    self.observations[current_ob][idx]['visits'] += 1
                else:
                    self.observations[current_ob][idx]['discounted_returns'][-1] = self.gamma * self.observations[current_ob][idx]['discounted_returns'][-1] + next_reward

            count -= 1
            self.set_optimal_state_action(current_ob, current_next_action)

        return
