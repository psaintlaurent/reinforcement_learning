"""
This needs major refactoring
The general idea is that this will eventually be a generic policy object
that is managed by a Policy Iteration Agent
"""
class Policy(object):
    observations, next_actions, reward = None, None, None
    returns, returns_count, optimal_state_action = None, None, None

    def __init__(self):
        self.observations, self.next_actions, self.reward = [], [], []
        self.returns, self.returns_count, self.state_action_value, self.optimal_state_action = {}, {}, {}, {}

    """
        This is a wildly inefficient method of keeping track of mappings of
        observations to actions but I'll worry about that later.
    """
    def add(self, observation, reward, next_action):
        if observation not in self.observations:
            self.observations.append(observation)
            self.next_actions.append([next_action])
            self.next_reward.append({(observation, next_action): reward})
        else:
            idx = self.observation_idx(observation)
            self.next_actions[idx].append(next_action)
            self.next_reward[idx].append({(observation, next_action): reward})

        return

    def observation_idx(self, observation):
        idx, policy_action = 0, None
        for ob in self.observations:
            if ob == observation:
                output = idx
                break
            idx += 1

        return output

    def set_optimal_state_action(self, observation_idx, action):

        if observation_idx not in self.optimal_state_action or \
                self.optimal_state_action[observation_idx]['return'] <= self.returns[(observation_idx, action)]:

            self.optimal_state_action[observation_idx] = {'action': action,
                                                          'return': self.policy.returns[(observation_idx, action)]}

    def next_action(self, observation):

        idx = self.get_observation_idx(observation)
        best_next_action, best_reward = None, None

        if idx in self.optimal_state_action:
            best_next_action, best_reward =

        if idx in self.next_reward:
            for tpl, reward in self.next_reward[idx].items():
                observation, next_action = tpl
                if next_action is None or reward > best_reward:
                    best_next_action, best_reward = next_action, reward

        return best_next_action, best_reward