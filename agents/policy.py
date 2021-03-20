"""
This needs major refactoring
The general idea is that this will eventually be a generic policy object
that is managed by a Policy Iteration Agent
"""
class Policy(object):
    observations, next_actions, reward = None, None, None

    def __init__(self):
        self.observations, self.next_actions, self.reward = [], [], []

    """
        This is a wildly inefficient method of keeping track of mappings of
        observations to actions but I'll worry about that later.
    """
    def add(self, observation, reward, next_action):
        if observation not in self.observations:
            self.observations.append(observation)
            self.next_actions.append([next_action])
            self.next_reward.append({next_action: reward})
        else:
            idx = self.observation_idx(observation)
            self.next_actions[idx].append(next_action)

        return

    def observation_idx(self, observation):
        idx, policy_action = 0, None
        for ob in self.observations:
            if ob == observation:
                output = idx
                break
            idx += 1

        return output

    def next_action(self, observation):

        idx = self.get_observation_idx(observation)
        best_next_action, best_reward = None, None

        if idx in self.next_reward:
            for next_action, reward in self.next_reward[idx].items():
                if next_action is None or reward > best_reward:
                    best_next_action, best_reward = next_action, reward

        return best_next_action, best_reward


    def act(self, observation, action_space):
        pass
