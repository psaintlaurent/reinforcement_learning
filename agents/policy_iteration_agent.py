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
    returns = None # These are the episodic discounted returns
    time_steps = None  # time steps per episode
    gamma = None  # discount rate

    def __init__(self):
        self.action_values = {}
        self.state_values = {}
        self.policy = policy.Policy()
        self.time_steps = 20
        self.gamma = .2
        self.returns = {}
        self.returns_count = {}
        self.state_action_value = {}

    def act(self, observation=None, reward=None, done=None):

        """
            If there is no existing policy then select an action at random.
            If there is an existing policy select an action according to the following policy.
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
    def initalize_policy(self, env):

        """ 
            Select the epsilon greedy action according to the policy.
            If no observation was found choose the next action at random. 
        """
        rng = default_rng(219215)
        random_number = rng.random()

        observation = env.reset()
        for idx in range(self.time_steps):

            """ Select an action """
            if random_number < self.epsilon or len(self.action_values) == 0: # Explore
                action = self.action_space.sample()
            elif random_number >= self.epsilon: # Exploit if there is an exploitable option

                action, _ = self.policy.next_action(current_state)
                if action is None:
                    action = self.action_space.sample()

            """ Take the selected action """
            observation, reward, done, _ = env.step(action)

            """ Add a new state -> action mapping to the existing policy """
            if current_state is not None:
                self.policy.add(current_state, reward, action)

            current_state, previous_reward, previous_action = observation, reward, action

        return


    def eval(self, env):

        observation = env.reset()

        self.initalize_policy(env)

        for observation_idx in range(reversed(self.time_steps-1)):

            action, reward = self.policy.next_action(observation_idx + 1)

            if (observation_idx, action) not in self.returns:
                self.returns[(observation_idx, action)] = 0

            self.returns[(observation_idx, action)] = self.gamma * self.returns[(action, reward)] + reward
            self.returns_count[(observation_idx, action)] += 1

            if observation_idx not in self.optimal_state_action:
                self.optimal_state_action[observation_idx] = 0
            elif self.optimal_state_action[observation_idx] <= self.returns[(observation_idx, action)]:
                self.optimal_state_action[observation_idx] = self.returns[(observation_idx, action)]


            """
            For tomorrow update the policy given the calculated optimal action:
            pi(a|St) <- 1 - epsilon + epsilon/|A(St)| if a = A*
                        epsilon/|A(St}|               if a <> A*
            """

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
