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
        p = self.sum_of_runs_each_arm[arm]
        p = p / (6 * self.times_each_arm[arm])
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
        while up - low > 1e-7:
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
            assert runs_scored < 7
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
            mean = self.sum_of_runs_each_arm[arm] / self.times_each_arm[arm]
        else:
            mean = 0
        shift = self.uncertainity_of_runs_each_arm[arm]
        return mean + shift

    def get_action(self, wicket, runs_scored):
        """
        action = np.random.randint(0, 6)
        self.actions.append(action)
        """
        # updating all parameters

        action = self.prev_action
        self.times_each_arm[action] += 1

        self.sum_of_wicket_each_arm[action] += wicket
        self.uncertainity_of_wicket_each_arm[action] = np.sqrt(
            2 * np.log(2 / self.delta) / self.times_each_arm[action]
        )

        self.sum_of_runs_each_arm[action] += runs_scored
        self.uncertainity_of_runs_each_arm[action] = np.sqrt(
            2 * np.log(2 / self.delta) / self.times_each_arm[action]
        )
        action = -1
        for i in range(self.num_of_arms):
            if self.upper_bound(action) < self.upper_bound(i):
                action = i
        self.prev_action = action
        self.actions.append(action)
        self.delta = 1 / (len(self.actions) + 1) ** 2
        return action


class EE21B001_Q2:
    def __init__(self):
        self.pred_mean = np.zeros(6, dtype=float)
        self.init_action = [i for i in range(6) for j in range(1)]
        self.Time = np.zeros((6,), dtype=int)
        self.UCB = np.zeros((6,), dtype=float)
        self.prev_action = -1
        self.i = 0

    def KL_divergence(self, p, q):
        if q == 0 and p == 0:
            return 0
        if q == 0 and not p == 0:
            return 1000000
        if q == 1 and p == 1:
            return 0
        if q == 1 and not p == 1:
            return 1000000
        if p == 0:
            return np.log(1 / (1 - q))
        if p == 1:
            return np.log(1 / q)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def KL_UCB(self, action, time):
        bound = np.log(1 + time * (np.log(time)) ** 2) / self.Time[action]
        pred_mean = self.pred_mean[action]
        a = pred_mean
        b = 1
        for i in range(10):
            mu = (a + b) / 2
            if self.KL_divergence(pred_mean, mu) < bound:
                a = mu
            else:
                b = mu
        return (a + b) / 2

    def get_action(self, wicket, runs_scored):
        reward = (runs_scored / 6) ** 10

        while self.i < 6:  # Playing Every Arm once
            if self.prev_action != -1:
                self.pred_mean[self.prev_action] += (
                    reward - self.pred_mean[self.prev_action]
                ) / self.Time[self.prev_action]
            ret = self.init_action[self.i]
            self.Time[ret] = self.Time[ret] + 1
            self.prev_action = ret
            self.i += 1
            return int(ret)

        self.pred_mean[self.prev_action] += (
            reward - self.pred_mean[self.prev_action]
        ) / self.Time[self.prev_action]
        # print(self.pred_mean)
        for j in range(6):
            self.UCB[j] = self.KL_UCB(j, self.i)
        ret = np.argmax(self.UCB)
        self.Time[ret] = self.Time[ret] + 1
        self.prev_action = ret
        self.i += 1
        return ret


class Checker:
    def __init__(self):
        self.arm_runs = np.zeros(6)
        self.arm_times_choosen = np.zeros(6)
        self.prev_arm = -1
        self.time = 0

    def UCB(self, avg_runs, picks):
        return avg_runs + np.sqrt(2.0 * np.log(self.time) / picks)

    def get_action(self, wicket, runs_scored):
        if self.time <= 5:
            self.arm_times_choosen[self.time] += 1

            if self.prev_arm != -1:
                self.arm_runs[self.prev_arm] += runs_scored
            self.prev_arm = self.time

        else:
            self.arm_runs[self.prev_arm] += runs_scored
            KL_values = []
            for arm in range(6):
                avg_runs = self.arm_runs[arm] / (self.arm_times_choosen[arm]) * 1.0
                value = self.UCB(avg_runs, self.arm_times_choosen[arm])
                KL_values.append(value)

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
        # for this example the regret of wickets is higher but the regret for the number of runs is reduced
        self.__p_out = np.array([0.1, 0.1, 0.2, 0.07, 0.1, 0.5])
        self.__p_run = np.array([0.01, 0.09, 0.85, 0.8, 0.1, 0.7])
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


balls = np.random.randint(1000) + 10


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
        reg += regret_s
    return reg / 10


print(comp(Random_Agent))
print(comp(UCB_Agent))
print(comp(KL_UCB))
print(comp(EE21B001_Q2))
print(comp(Checker))

agent = KL_UCB()
environment = Environment(balls, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)

#
agent = UCB_Agent()
environment = Environment(balls, agent)
(
    regret_w,
    regret_s,
    regret_rho,
    total_runs,
    total_wickets,
    run_time,
) = environment.innings()
print(regret_w, regret_s, regret_rho, total_runs, total_wickets, run_time)
