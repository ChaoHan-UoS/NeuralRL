class EnvSampler():
    def __init__(self, env):
        self.env = env
        self.current_state = None
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()[0]

        cur_state = self.current_state
        if agent is None:
            action = self.env.action_space.sample()
        else:
            action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        done = terminated or truncated
        if done:
            self.current_state = None
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, done, info
