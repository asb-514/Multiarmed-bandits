import numpy as np


class ROLL_NUMBER_UCB:
    def __init__(self, num_arms, num_samples, sigma):
        self.__num_arms = num_arms
        self.__num_samples = num_samples
        self.__sigma = sigma

    def get_action(self, reward):
        # write your code here and return the action
        return np.random.randint(
            0, num_arms
        )  # for now I have implemented a dummy agent


class EE21B021_UCB:
    def __init__(self, num_arms, num_samples, sigma):
        self.__num_arms = num_arms
        self.__num_samples = num_samples
        self.__sigma = sigma
        self.myarmreward = np.zeros((self.__num_arms,), dtype=float)
        self.myarmepsilon = np.zeros((self.__num_arms,), dtype=float)
        for i in range(len(self.myarmepsilon)):
            self.myarmepsilon[i] = 1e9  # initial uncertainty is high
        self.myarmtimes = np.zeros((self.__num_arms,), dtype=int)
        self.mytime = 0
        self.myprevarm = -1

    def get_action(self, reward):
        if self.mytime != 0:
            self.myarmreward[self.myprevarm] += reward
        self.mytime += 1
        for i in range(len(self.myarmtimes)):
            if self.myarmtimes[i] < 1:
                self.myprevarm = i
                self.myarmtimes[i] += 1
                return i
        for i in range(self.__num_arms):
            self.myarmepsilon[i] = np.sqrt(
                2 * np.log(2 * ((self.mytime) ** 2)) / (self.myarmtimes[i])
            )
        bestarm = -1e9
        bestmean = -1e9
        for i in range(self.__num_arms):
            if (
                bestmean
                < (self.myarmreward[i] / self.myarmtimes[i]) + self.myarmepsilon[i]
            ):
                bestmean = (
                    self.myarmreward[i] / self.myarmtimes[i]
                ) + self.myarmepsilon[i]
                bestarm = i
        assert bestarm >= 0
        self.myprevarm = bestarm
        self.myarmtimes[bestarm] += 1
        return bestarm


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
    agent = EE21B021_UCB(num_arms, num_samples, sigma)
    env = Environment(num_arms, mean, sigma, agent)
    action_list = env.run()
    # print(mean)
    # print(agent.myarmreward/agent.myarmtimes)
    # print(agent.myarmepsilon)
    # print(agent.myarmreward)
    # print(agent.myarmtimes)
    regret = 0.0
    for i in range(num_arms):
        regret += (sorted_mean[-1] - mean[i]) * (agent.myarmtimes[i])

    avg_reg += regret

print(avg_reg / 1000)
