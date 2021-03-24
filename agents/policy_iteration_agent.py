import argparse
import gym
import sys
from . import policy
from gym import wrappers, logger
from numpy.random import default_rng

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

    def __init__(self):
        self.action_values = {}
        self.state_values = {}
        self.policy = policy.Policy()
        self.time_steps = 20
        self.gamma = .2


    def act(self, observation=None, reward=None, done=None):

        """
            If there is no existing policy then select an action at random.
            If there is an existing policy select an action according to the policy
        """
        if len(self.policy) == 0:
            output = self.actions_space.sample()
        else:
            output, _ = self.policy.next_action(observation)

        return output

    """
        Initialize the policy by generating an episode following pi, 
        an initially random epsilon soft policy.
        The code for this is going to look odd.
    """
    def initialize_policy(self, env):

        """ 
            Select the epsilon greedy action according to the policy.
            If no observation was found choose the next action at random. 
        """
        rng = default_rng(219215)
        random_number = rng.random()

        observation = env.reset()
        for idx in range(self.time_steps):

            """ Select an action """
            if random_number < self.epsilon or len(self.action_values) == 0:  # Explore
                action = env.action_space.sample()
            elif random_number >= self.epsilon:  # Exploit if there is an exploitable option

                action, _ = self.policy.next_action(current_state)
                if action is None:
                    action = env.action_space.sample()

            """ Take the selected action """
            observation, reward, done, _ = env.step(action)

            """ Add a new state -> action mapping to the existing policy """
            if current_state is not None:
                self.policy.add(current_state, reward, action)

            """ Store the existing state, reward and action """
            current_state = observation
            previous_reward = reward
            previous_action = action

        return

    """

    """
    def eval(self, env):

        observation = env.reset()
        self.initialize_policy(env)

        """
        The number of time steps and the number of observations should be equal.
        """
        count = self.time_steps - 1
        ob_iter = reversed(self.policy.observations.keys())
        discounted_return = 0
        return_count = 0

        """
        TODO extrapolate to N-step
        """

        while count > 0:

            next_ob = self.policy.observations(next(ob_iter))
            current_ob = self.policy.observations(next(ob_iter))

            current_action, current_reward = self.policy.next_action(current_ob)
            next_action, next_reward = self.policy.next_action(next_ob)

            """ 
                Calculate the first visit return
            """
            discounted_return = self.gamma * discounted_return + next_reward

            for idx in self.observations[current_ob]:
                self.observations[current_ob][idx]['next_action']


            count -= 1




            self.policy.returns[(observation_idx, action)] = self.gamma * self.policy.returns[(observation_idx, action)] + reward
            self.policy.returns_count[(observation_idx, action)] += 1

            """
                Set the optimal state action
            """
        self.policy.set_optimal_state_action(observation_idx, action)


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

    agent = PolicyIterationAgent(env.action_space)

    env.close()
