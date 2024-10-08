import numpy as np


class Random_ETC_DELTA:
    def __init__(self, num_arms, num_samples, sigma, Delta):
        self.__num_arms = num_arms
        self.__num_samples = num_samples
        self.__sigma = sigma
        self.__Delta = Delta

    def get_action(self, reward):
        # write your code here and return the action
        return np.random.randint(
            0, num_arms
        )  # for now I have implemented a dummy agent


class EE21B021_ETC_DELTA:
    def __init__(self, num_arms, num_samples, sigma, Delta):
        self.__num_arms = num_arms
        self.__num_samples = num_samples
        self.__sigma = sigma
        self.__Delta = Delta
        self.myarmreward = np.zeros(num_arms, dtype=float)
        self.mytimes = np.zeros(num_arms, dtype=float)
        self.mym = 20  # have to change
        # self.mym =int(8*np.log(2/0.3)/(Delta**2))
        self.myfirstplay = np.array([], dtype=int)
        self.start = 0
        self.myprev_action = -1
        self.mybestarm = -1
        for i in range(self.__num_arms):
            for j in range(self.mym):
                self.myfirstplay = np.append(self.myfirstplay, int(i))

    def get_action(self, reward):
        if self.start != 0:
            self.myarmreward[self.myprev_action] += reward

        self.start = 1
        if len(self.myfirstplay) != 0:
            temp = self.myfirstplay[-1]
            self.myfirstplay = self.myfirstplay[:-1]
            self.myprev_action = temp
            self.mytimes[temp] += 1
            return temp

        if self.mybestarm == -1:
            mean = -1e9
            for i in range(self.__num_arms):
                if mean < self.myarmreward[i] / self.mytimes[i]:
                    mean = self.myarmreward[i] / self.mytimes[i]
                    self.mybestarm = i
            self.myprev_action = self.mybestarm

        self.mytimes[self.mybestarm] += 1
        return self.mybestarm


class Environment:
    def __init__(self, num_arms, mean, sigma, agent):
        self.__num_samples = num_samples
        self.__num_arms = num_arms
        self.__mean = mean
        self.__reward = -100
        self.__action = np.zeros((num_samples,), dtype=int)
        self.__agent = agent
        self.__regret = 0

    def run(self):
        for i in range(self.__num_samples):
            self.__action[i] = self.__agent.get_action(self.__reward)
            self.__reward = self.__mean[self.__action[i]] + sigma * np.random.randn()
        return self.__action


# num_samples = 100
# num_arms = 4
# mean = np.random.rand(
#     num_arms,
# )
# sigma = 1
# sorted_mean = np.sort(mean)
# Delta = sorted_mean[-1] - sorted_mean[-2]
# agent = EE21B021_ETC_DELTA(num_arms, num_samples, sigma, Delta)
# env = Environment(num_arms, mean, sigma, agent)
# action_list = env.run()
# print(mean)
# print(agent.myarmreward/agent.mytimes)
# print(agent.myarmreward)
# print(agent.mytimes)
# print(action_list)
avg_reg = 0.0
for _ in range(1000):
    num_samples = 100
    num_arms = 4
    mean = np.random.rand(
        num_arms,
    )
    sigma = 5
    sorted_mean = np.sort(mean)
    Delta = sorted_mean[-1] - sorted_mean[-2]
    agent = EE21B021_ETC_DELTA(num_arms, num_samples, sigma, Delta)
    env = Environment(num_arms, mean, sigma, agent)
    action_list = env.run()
    # print(mean)
    # print(agent.myarmreward/agent.mytimes)
    # print(agent.myarmreward)
    # print(agent.mytimes)
    regret = 0.0
    for i in range(num_arms):
        regret += (sorted_mean[-1] - mean[i]) * (agent.mytimes[i])

    avg_reg += regret

print(avg_reg / 1000)
