import time
import numpy as np


class Random_Agent:
    def __init__(self):
        self.prev_action = -1
        self.times_each_arm = []
        self.sum_of_wicket_each_arm = []
        self.sum_of_runs_each_arm = []
        self.uncertainity_of_runs_each_arm = []
        self.uncertainity_of_wicket_each_arm = []
        ## change based on number of arms to pick
        self.num_of_arms = 6
        ## tinker to get best results
        self.delta = 1
        for _ in range(self.num_of_arms):
            self.times_each_arm.append(0)
            self.sum_of_runs_each_arm.append(0)
            self.sum_of_wicket_each_arm.append(0)
            self.uncertainity_of_runs_each_arm.append(
                1e18
            )  # inilitally infinite uncertainity
            self.uncertainity_of_wicket_each_arm.append(
                1e18
            )  # inilitally infinite uncertainity
        pass

    def get_action(self, wicket, runs_scored):
        action = self.prev_action
        self.times_each_arm[action] += 1
        self.sum_of_wicket_each_arm[action] += wicket
        self.sum_of_runs_each_arm[action] += runs_scored
        action = np.random.randint(0, 6)
        self.prev_action = action
        return action


class UCB_Agent:
    def __init__(self):
        self.actions = []
        self.total_wickets = 0
        self.prev_action = -1
        self.times_each_arm = []
        self.sum_of_wicket_each_arm = []
        self.sum_of_runs_each_arm = []
        self.uncertainity_of_runs_each_arm = []
        self.uncertainity_of_wicket_each_arm = []
        ## change based on number of arms to pick
        self.num_of_arms = 6
        ## tinker to get best results
        self.delta = 1
        self.time = 0
        for _ in range(self.num_of_arms):
            self.times_each_arm.append(0)
            self.sum_of_runs_each_arm.append(0)
            self.sum_of_wicket_each_arm.append(0)
            self.uncertainity_of_runs_each_arm.append(
                1e18
            )  # inilitally infinite uncertainity
            self.uncertainity_of_wicket_each_arm.append(
                1e18
            )  # inilitally infinite uncertainity
        pass

    def upper_bound(self, arm):
        """returns the upper bound in the confidence inteval

        :arm: arm being played
        """
        if arm == -1:
            return -1e18
        # we should pick arm with minimum wicket, so its same as picking the maximum of the -ve value of the wickets
        if self.times_each_arm[arm] != 0:
            meanw = self.sum_of_wicket_each_arm[arm] / self.times_each_arm[arm]
        else:
            meanw = 0
        if self.times_each_arm[arm] != 0:
            meanr = self.sum_of_runs_each_arm[arm] / self.times_each_arm[arm]
        else:
            meanr = 0
        if meanw != 0:
            mean = meanr / meanw
        else:
            mean = 0
        # if not played this arm keep playing
        # if the wicket has not fallen then keep playing that arm
        if meanw == 0:
            shift = 1e18
        else:
            shift = np.sqrt(
                2 * np.log(1 / self.delta) / self.sum_of_wicket_each_arm[arm]
            )

        return mean + shift

    def get_action(self, wicket, runs_scored):
        """
        action = np.random.randint(0, 6)
        self.actions.append(action)
        """
        # updating all parameters

        self.time += 1
        self.delta = 1 / (self.time + 1) ** 2
        if self.time > 1:
            action = self.prev_action
            self.times_each_arm[action] += 1

            self.sum_of_wicket_each_arm[action] += wicket
            if wicket == 1:
                self.total_wickets += 1
            self.uncertainity_of_wicket_each_arm[action] = np.sqrt(
                2 * np.log(2 / self.delta) / self.times_each_arm[action]
            )
            self.sum_of_runs_each_arm[action] += runs_scored
            self.uncertainity_of_runs_each_arm[action] = np.sqrt(
                2 * np.log(2 / self.delta) / self.times_each_arm[action]
            )
        action = -1
        for i in range(1, self.num_of_arms):
            if self.upper_bound(action) <= self.upper_bound(i):
                action = i
        self.prev_action = action
        self.actions.append(action)
        return action


class ROLLNUMBER_Q1:
    def __init__(self):
        self.__num_arms = 6

        self.__counts = np.zeros(self.__num_arms)

        # self.__values = np.zeros(self.__num_arms)

        # self.__costs = np.zeros(self.__num_arms)

        self.__values = 1e-8 * np.ones(self.__num_arms)

        self.__costs = 1e-8 * np.ones(self.__num_arms)

        self.__values_seq = [np.array([]) for _ in range(self.__num_arms)]

        self.__costs_seq = [np.array([]) for _ in range(self.__num_arms)]

        self.__dummy = 1e-8 * np.ones(self.__num_arms)

        self.__prev_arm = -1

        self.__count = 0

        self.__alpha = 5

        self.__m_r = 6

        self.__m_x = 1

        self.__l = 2

    def __compute_covariance(self, x_samples, y_samples):
        # Check if the sample sizes are equal

        """if x_samples.shape[0] != y_samples.shape[0]:

        raise ValueError("Sample sizes are not equal")"""

        n = x_samples.shape[0]

        # Compute the means of x and y

        mean_x = np.mean(x_samples)

        mean_y = np.mean(y_samples)

        # Compute the covariance

        covariance = np.sum((x_samples - mean_x) * (y_samples - mean_y)) / (n)

        return covariance

    def __compute_variance(self, samples):
        # Compute the mean of the samples

        mean = np.mean(samples)

        # Compute the squared differences from the mean

        squared_diff = (samples - mean) ** 2

        # Compute the variance

        variance = np.sum(squared_diff) / (samples.shape[0])

        return variance

    def get_action(self, wicket, runs_scored):
        self.__count += 1

        if self.__count > 1:
            self.__counts[self.__prev_arm] += 1.0

            self.__values_seq[self.__prev_arm] = np.append(
                self.__values_seq[self.__prev_arm], runs_scored
            )

            self.__costs_seq[self.__prev_arm] = np.append(
                self.__costs_seq[self.__prev_arm], wicket
            )

            self.__values[self.__prev_arm] += (
                runs_scored - self.__values[self.__prev_arm]
            ) / self.__counts[self.__prev_arm]

            # print(self.__values)

            self.__costs[self.__prev_arm] += (
                wicket - self.__costs[self.__prev_arm]
            ) / self.__counts[self.__prev_arm]

        # print(self.__prev_arm , self.__values[self.__prev_arm])

        # exploration phase

        if self.__count <= self.__num_arms:
            self.__prev_arm += 1

            return self.__prev_arm

        n = self.__count

        r = np.zeros(self.__num_arms)

        w = np.zeros(self.__num_arms)

        v = np.zeros(self.__num_arms)

        var = np.zeros(self.__num_arms)

        epsilon = np.zeros(self.__num_arms)

        eta = np.zeros(self.__num_arms)

        c = np.zeros(self.__num_arms)

        ucb_b1 = np.zeros(self.__num_arms)

        for arm in range(self.__num_arms):
            r[arm] = max(self.__dummy[arm], self.__values[arm]) / max(
                self.__dummy[arm], self.__costs[arm]
            )

            w[arm] = self.__compute_covariance(
                self.__values_seq[arm], self.__costs_seq[arm]
            ) / max(self.__compute_variance(self.__costs_seq[arm]), self.__dummy[arm])

            v[arm] = self.__compute_variance(self.__values_seq[arm]) - w[
                arm
            ] ** 2 * self.__compute_variance(self.__costs_seq[arm])

            var[arm] = self.__compute_variance(self.__costs_seq[arm])

            epsilon[arm] = (
                2 * self.__alpha * self.__m_r * np.log(n) / (3 * self.__counts[arm])
                + (self.__l * self.__alpha * v[arm] * np.log(n) / self.__counts[arm])
                ** 0.5
            )

            eta[arm] = (
                2 * self.__alpha * self.__m_x * np.log(n) / (3 * self.__counts[arm])
                + (self.__l * self.__alpha * var[arm] * np.log(n) / self.__counts[arm])
                ** 0.5
            )

            # print(arm,self.__values[arm])

            c[arm] = (
                1.4
                * (epsilon[arm] + (r[arm] - w[arm]) * eta[arm])
                / max(self.__dummy[arm], self.__costs[arm])
            )

            ucb_b1[arm] = r[arm] + c[arm]

        """ for arm in range(self.__num_arms):

      ucb_b1[arm] = self.__compute(arm) """

        self.__prev_arm = np.argmax(ucb_b1)

        # print(self.__prev_arm)

        # print(self.__prev_arm , self.__values[self.__prev_arm])

        # action = np.random.randint(0,6)

        # for my reference

        # print(self.__counts)

        return self.__prev_arm


class KL_UCB:
    def __init__(self):
        self.arm_runs = np.zeros(6)
        self.arm_wickets = np.zeros(6)
        self.arm_times_choosen = np.zeros(6)
        self.prev_arm = -1
        self.time = 0

    def KL_dig(self, p, q):
        if p == 0 and q == 0:
            return 0
        elif q == 0 and p != 0:
            return 10**4
        elif q == 1 and p != 1:
            return 10**4
        elif p == 1 and q == 1:
            return 0
        elif p == 1:
            return np.log(1.0 / q)
        elif p == 0:
            return np.log(1.0 / (1 - q))
        else:
            return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def KL_UCB(self, avg_runs, picks):
        high, low = 1, avg_runs
        precision = 10 ** (-5)
        f_t = np.log(1 + self.time * (np.log(self.time)) ** 2)
        while high - low > precision:
            mid = (high + low) / 2.0
            if self.KL_dig(avg_runs, mid) > f_t / picks:
                high = mid
            else:
                low = mid
        return high

    def get_action(self, wicket, runs_scored):
        runs_scored = runs_scored
        wicket = 1 - wicket
        if self.time <= 5:
            self.arm_times_choosen[self.time] += 1
            if self.prev_arm != -1:
                self.arm_runs[self.prev_arm] += runs_scored
                self.arm_wickets[self.prev_arm] += wicket
            self.prev_arm = self.time

        else:
            self.arm_runs[self.prev_arm] += runs_scored
            self.arm_wickets[self.prev_arm] += wicket
            KL_values = []
            for arm in range(6):
                avg_runs = self.arm_runs[arm] / (self.arm_times_choosen[arm]) * 1.0
                avg_wickets = (
                    self.arm_wickets[arm] / (self.arm_times_choosen[arm]) * 1.0
                )
                value1 = self.KL_UCB(avg_runs, self.arm_times_choosen[arm])
                value2 = self.KL_UCB(avg_wickets, self.arm_times_choosen[arm])
                KL_values.append(value1 * value2)

            optimal_arm = np.argmax(KL_values)
            self.arm_times_choosen[optimal_arm] += 1
            self.prev_arm = optimal_arm

        self.time += 1
        return self.prev_arm


class Environment:
    def __init__(self, num_balls, agent):
        self.num_balls = num_balls
        self.agent = agent
        self.__run_time = 0
        self.__total_runs = 0
        self.__total_wickets = 0
        self.__runs_scored = 0
        self.__start_time = 0
        self.__end_time = 0
        self.__regret_w = 0
        self.__regret_s = 0
        self.__wicket = 0
        self.__regret_rho = 0
        self.__p_out = np.array([0.001, 0.01, 0.02, 0.03, 0.1, 0.3])
        self.__p_run = np.array([1, 0.9, 0.85, 0.8, 0.75, 0.7])
        self.__action_runs_map = np.array([0, 1, 2, 3, 4, 6])
        self.__s = (1 - self.__p_out) * self.__p_run * self.__action_runs_map
        self.__rho = self.__s / self.__p_out
        # print(self.__rho)

    def __get_action(self):
        self.__start_time = time.time()
        action = self.agent.get_action(self.__wicket, self.__runs_scored)
        self.__end_time = time.time()
        self.__run_time = self.__run_time + self.__end_time - self.__start_time
        return action

    def __get_outcome(self, action):
        pout = self.__p_out[action]
        prun = self.__p_run[action]
        wicket = np.random.choice(2, 1, p=[1 - pout, pout])[0]
        runs = 0
        if wicket == 0:
            runs = (
                self.__action_runs_map[action]
                * np.random.choice(2, 1, p=[1 - prun, prun])[0]
            )
        return wicket, runs

    def innings(self):
        self.__total_runs = 0
        self.__total_wickets = 0
        self.__runs_scored = 0

        for ball in range(self.num_balls):
            action = self.__get_action()
            self.__wicket, self.__runs_scored = self.__get_outcome(action)
            self.__total_runs = self.__total_runs + self.__runs_scored
            self.__total_wickets = self.__total_wickets + self.__wicket
            self.__regret_w = self.__regret_w + (
                self.__p_out[action]
                - np.min(self.__p_out)  # mywicket - minpossibelwicket
            )
            self.__regret_s = self.__regret_s + (
                np.max(self.__s) - self.__s[action]
            )  # max run - my run
            self.__regret_rho = self.__regret_rho + (
                np.max(self.__rho) - self.__rho[action]
            )
        return (
            self.__regret_w,
            self.__regret_s,
            self.__regret_rho,
            self.__total_runs,
            self.__total_wickets,
            self.__run_time,
        )


## here the wickets act as ball in previous case, before we optimized runs/balls, now we should optimize runs/wickets
agent = Random_Agent()
environment = Environment(100, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)
# print(agent.sum_of_wicket_each_arm)
# print(agent.times_each_arm)

agent = UCB_Agent()
environment = Environment(100, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)

agent = ROLLNUMBER_Q1()
environment = Environment(100, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)
# print(environment.__rho)


balls = np.random.randint(1000) + 100


def comp(agents):
    reg = 0
    for _ in range(10):
        agent = agents()
        environment = Environment(balls, agent)
        (
            regret_w,
            regret_s,
            regret_rho,
            total_runs,
            total_wickets,
            run_time,
        ) = environment.innings()
        reg += regret_rho
    return reg / 10


print(comp(Random_Agent))
print(comp(ROLLNUMBER_Q1))
print(comp(UCB_Agent))
print(comp(KL_UCB))
