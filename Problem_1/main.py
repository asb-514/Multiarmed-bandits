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


class KL_UCB:
    def __init__(self):
        self.prev_action = -1
        self.times_each_arm = []
        self.sum_of_wicket_each_arm = []
        self.sum_of_runs_each_arm = []
        ## change based on number of arms to pick
        self.num_of_arms = 6
        self.time = 0
        for _ in range(self.num_of_arms):
            self.times_each_arm.append(0)
            self.sum_of_runs_each_arm.append(0)
            self.sum_of_wicket_each_arm.append(0)
        pass

    def compute(self, arm):
        # p = self.sum_of_runs_each_arm[arm]
        # p = p / (6 * self.times_each_arm[arm])
        p = 1 - (self.sum_of_wicket_each_arm[arm] / self.times_each_arm[arm])
        # p in between one now
        DIVMAX = 1e10

        def kl_div(p, q):
            if q == 0 and p == 0:
                return 0
            elif q == 0 and p != 0:
                return DIVMAX
            elif q == 1 and p == 1:
                return 0
            elif q == 1 and p != 1:
                return DIVMAX
            elif p == 0:
                return np.log(1 / (1 - q))
            elif p == 1:
                return np.log(1 / q)
            else:
                return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

        low = p
        up = 1
        while up - low > 1e-10:
            mid = (low + up) / 2.0
            if (
                kl_div(p, mid)
                <= np.log(1 + self.time * (np.log(self.time) ** 2))
                / self.times_each_arm[arm]
            ):
                low = mid
            else:
                up = mid

        return low

    def get_action(self, wicket, runs_scored):
        self.time += 1
        if self.time > 1:
            action = self.prev_action
            self.sum_of_wicket_each_arm[action] += wicket
            self.sum_of_runs_each_arm[action] += runs_scored

        for i in range(self.num_of_arms):
            if self.times_each_arm[i] == 0:
                self.prev_action = i
                self.times_each_arm[i] += 1
                return i
        action = -1
        val = -1
        for i in range(self.num_of_arms):
            if val < self.compute(i):
                val = self.compute(i)
                action = i
        assert action != -1
        self.prev_action = action
        self.times_each_arm[action] += 1
        return action


class UCB_Agent:
    def __init__(self):
        self.actions = []
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
        self.time = 1
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
            mean = -1 * (self.sum_of_wicket_each_arm[arm] / self.times_each_arm[arm])
        else:
            mean = 0
        shift = self.uncertainity_of_wicket_each_arm[arm]
        return mean + shift

    def get_action(self, wicket, runs_scored):
        """
        action = np.random.randint(0, 6)
        self.actions.append(action)
        """
        # updating all parameters
        self.delta = 1 / self.time**2
        self.time += 1
        action = self.prev_action
        self.times_each_arm[action] += 1

        self.sum_of_wicket_each_arm[action] += wicket
        self.uncertainity_of_wicket_each_arm[action] = np.sqrt(
            2 * np.log(1 / self.delta) / self.times_each_arm[action] ** 2
        )

        self.sum_of_runs_each_arm[action] += runs_scored
        self.uncertainity_of_runs_each_arm[action] = np.sqrt(
            2 * np.log(1 / self.delta) / self.times_each_arm[action] ** 2
        )

        action = -1
        for i in range(self.num_of_arms):
            if self.upper_bound(action) < self.upper_bound(i):
                action = i
        self.prev_action = action
        return action


class ROLLNUMBER_Q1:
    def __init__(self):
        self.__num_arms = 6

        self.__counts = np.zeros(self.__num_arms)

        self.__values = np.zeros(self.__num_arms)

        self.__prev_arm = -1

        self.__count = 0

        # self.__prev_

    def __compute(self, arm):
        # kl divergence formulae

        def kl_divergence(p, q):
            if q == 0 and p == 0:
                return 0

            elif q == 0 and not p == 0:
                return DIV_MAX

            elif q == 1 and p == 1:
                return 0

            elif q == 1 and not p == 1:
                return DIV_MAX

            elif p == 0:
                return np.log(1 / (1 - q))

            elif p == 1:
                return np.log(1 / q)

            return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

        DIV_MAX = 10000000

        precision = 1e-30

        max_iter = 50

        n = 0

        p = self.__values[arm]

        lower_bound = self.__values[arm]

        upper_bound = 1

        # loop to find the best q, i.e ucb

        while n < max_iter and upper_bound - lower_bound > precision:
            q = (lower_bound + upper_bound) / 2

            if (
                kl_divergence(p, q)
                > np.log(1 + self.__count * (np.log(self.__count) ** 2))
                / self.__counts[arm]
            ):
                upper_bound = q

            else:
                lower_bound = q

            n += 1

        result = lower_bound

        return result

    def get_action(self, wicket, runs_scored):
        self.__count += 1

        if self.__count > 1:
            self.__counts[self.__prev_arm] += 1

            self.__values[self.__prev_arm] += (
                2 - wicket - self.__values[self.__prev_arm]
            ) / self.__counts[self.__prev_arm]

        # print(self.__prev_arm , self.__values[self.__prev_arm])

        # exploration phase

        if self.__count <= self.__num_arms:
            self.__prev_arm += 1

            return self.__prev_arm

        kcl_ucb = np.zeros(self.__num_arms)

        for arm in range(self.__num_arms):
            kcl_ucb[arm] = self.__compute(arm)

        self.__prev_arm = np.argmax(kcl_ucb)

        # print(self.__prev_arm , self.__values[self.__prev_arm])

        # action = np.random.randint(0,6)

        return self.__prev_arm


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
        self.__p_out = np.array([0.01, 0.1, 0.2, 0.03, 0.01, 1])
        self.__p_run = np.array([1, 0.9, 0.85, 0.8, 0.75, 0.7])
        # self.__p_out = np.array([(np.random.randint(300) + 1) / 1000 for _ in range(6)])
        # self.__p_run = np.array([(np.random.randint(300) + 1) / 1000 for _ in range(6)])
        self.__action_runs_map = np.array([0, 1, 2, 3, 4, 6])
        self.__s = (1 - self.__p_out) * self.__p_run * self.__action_runs_map
        self.__rho = self.__s / self.__p_out

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


balls = np.random.randint(100) + 10


def comp(agents):
    reg = 0
    for i in range(10):
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
        reg += regret_w
    return reg / 10


print(comp(Random_Agent))
print(comp(KL_UCB))
print(comp(ROLLNUMBER_Q1))

agent = ROLLNUMBER_Q1()
environment = Environment(balls, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
# print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)

# print(agent.sum_of_wicket_each_arm)
# print(agent.times_each_arm)
# print(agent.actions)
# for i in range(6):
#     print(i, " ", agent.upper_bound(i))
