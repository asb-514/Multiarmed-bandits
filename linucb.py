import numpy as np


class ROLL_NUMBER_LINUCB:
    def __init__(self, num_dim, num_samples, sigma):
        self.__num_dim = num_dim
        self.__num_samples = num_samples
        self.__sigma = sigma

    def get_action(self, reward):
        # write your code here and return the action
        return np.random.rand(
            num_dim,
        )  # for now I have implemented a dummy agent


class EE21B021_NUMBER_LINUCB:
    def __init__(self, num_dim, num_samples, sigma, arm):
        self.__num_dim = num_dim
        self.__num_samples = num_samples
        self.__sigma = sigma
        self.__arms = arm
        self.myV = np.array([0] * self.__num_dim * self.__num_dim, dtype=float).reshape(
            self.__num_dim, self.__num_dim
        )
        self.lamda = 0.001
        self.myV = self.lamda * np.identity(self.__num_dim)
        # print(self.myV)
        self.time = 0
        self.esttheta = np.array([0] * self.__num_dim, dtype=float)
        self.myb = np.array([0] * self.__num_dim, dtype=float)
        self.myarmtimes = np.array([0] * len(self.__arms), dtype=float)
        self.prevarm = -1

    def get_action(self, reward):
        if self.time != 0:
            self.myb = self.myb + reward * self.__arms[self.myprevarm]
            dumb = np.copy(self.myb)
            dumb = dumb.reshape(self.__num_dim, 1)
            self.esttheta = np.matmul(np.linalg.inv(self.myV), dumb)
            self.esttheta = self.esttheta.reshape(self.__num_dim)
        now = 0
        nowbest = -1e9
        self.time += 1
        for i in range(len(self.__arms)):
            if self.myarmtimes[i] <= 1:
                now = i
                self.myarmtimes[now] += 1
                self.myprevarm = now
                a = np.copy(self.__arms[now])
                a = a.reshape(self.__num_dim, 1)
                self.myV = self.myV + np.matmul(a, np.transpose(a))
                return now
        beta = np.sqrt(self.lamda * self.__num_dim) + np.sqrt(
            2 * np.log(self.time**1)
            + self.__num_dim * np.log(1 + (self.time / (self.lamda)))
        )
        for i in range(len(self.__arms)):
            a = np.copy(self.__arms[i])
            a = a.reshape(self.__num_dim, 1)
            theta = np.copy(self.esttheta)
            theta.reshape(self.__num_dim, 1)
            term = np.matmul(np.linalg.inv(self.myV), a)
            term = np.matmul(np.transpose(a), term)
            cur = np.matmul(np.transpose(a), theta) + beta * np.sqrt(term)
            if cur > nowbest:
                now = i
                nowbest = cur
        self.myarmtimes[now] += 1
        self.myprevarm = now
        # print(np.matmul(np.transpose(self.__arms[now]), self.__arms[now]))
        # print(np.matmul((self.__arms[now]),np.transpose(self.__arms[now])))
        a = np.copy(self.__arms[now])
        a = a.reshape(self.__num_dim, 1)
        # print(a)
        # print(np.matmul(a,np.transpose(a)))
        self.myV = self.myV + np.matmul(a, np.transpose(a))
        return now


class Environment:
    def __init__(self, theta_star, mean, sigma, agent):
        self.__num_samples = num_samples
        self.__theta_star = theta_star
        self.__mean = mean
        self.__reward = -100
        self.__action = np.zeros((num_samples, num_dim), dtype=float)
        self.__agent = agent
        self.__regret = 0

    def run(self):
        for i in range(self.__num_samples):
            self.__action[i, :] = self.__agent.get_action(self.__reward)
            self.__reward = (
                np.dot(self.__action[i, :], theta_star) + sigma * np.random.randn()
            )
        return self.__action


num_samples = 1000
num_arms = 4
num_dim = 4
theta_star = 2 * (
    np.random.rand(
        num_dim,
    )
    - 0.5
)
arm = []
for i in range(num_arms):
    curarm = []
    for j in range(num_dim):
        t = np.random.rand()
        t -= 0.5
        t *= 2
        curarm.append(t)
    arm.append(curarm)
arm = np.array(arm, dtype=float)
sigma = 1
# agent = ROLL_NUMBER_LINUCB(num_dim, num_samples, sigma)
# env = Environment(num_dim, theta_star, sigma, agent)
# action_list = env.run()
print(theta_star)
agent = EE21B021_NUMBER_LINUCB(num_dim, num_samples, sigma, arm)
env = Environment(num_dim, theta_star, sigma, agent)
action_list = env.run()
print(agent.esttheta)
print(agent.myarmtimes)
for i in range(num_arms):
    print(np.dot(arm[i], theta_star))
