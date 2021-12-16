import random as rd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import networkmodel
from networkmodel import Iterate


class param:
    def __init__(self, p, iter_setting, true_ans):
        self.p = p
        self.true_ans = true_ans
        self.score = 0
        self.p_for_iter = iter_setting
        self.ans = []
        self.update_score()

    def update_score(self):
        # print('Updaing score')
        netinfo = Iterate(self.p, self.p_for_iter[0], self.p_for_iter[1])
        avgdeg = netinfo.iloc[-1, 5]
        Ng = netinfo.iloc[-1, 7]  ## of nodes of the largest cluster
        self.ans = [avgdeg, Ng]
        self.score = math.sqrt(
            (self.ans[0] - self.true_ans[0]) ** 2
            + (self.ans[1] - self.true_ans[1]) ** 2
        )
        # print('Ending Update')


class GA:
    def __init__(
        self,
        iter_setting,
        true_ans,
        limit1=1048575,
        limit2=1048575,
        level=100,
        populations=50,
        crossover_times=20,
        crossover_length=5,
        mating_pool_num=40,
        mutate=0.02,
    ):
        self.iter_setting = (
            iter_setting  # a list, [Initialized total nodes, # of Iteration]
        )
        self.true_ans = true_ans
        self.limit1 = limit1  # c1上限
        self.limit2 = limit2  # c2上限
        self.level = level  # 跑幾次
        self.populations = populations
        self.crossover_times = crossover_times  # 每一次iteration時，crossover的次數
        self.crossover_length = crossover_length
        self.mating_pool_num = mating_pool_num  # mating pool中的數量
        self.mutate = mutate  # mutate的機率

    def binary_to_decimal(self, bi):
        return int(str(bi), 2)

    def decimal_to_binary(self, dec):
        return int("{0:b}".format(dec))

    # 隨機生成最初的populations
    def _init_Params(self):
        print("Start to Initialize")
        Params = []
        upper_bound_c1 = self.limit1
        upper_bound_c2 = self.limit2
        x = 0
        # 在到達我們要的populations的數目之前，不斷生成亂數
        while x < self.populations:
            candidate_c1 = rd.randrange(0, upper_bound_c1)
            candidate_c2 = rd.randrange(0, upper_bound_c2)
            Params.append(
                param(
                    [candidate_c1 / 1000000, candidate_c2 / 1000000],
                    self.iter_setting,
                    self.true_ans,
                )
            )
            x += 1
        return Params

    # param = [c1, c2] -> binary(c1)和 binary(c2)並在一起
    def change_to_code(self, param):
        large_param = [int(param[0] * 1000000), int(param[1] * 1000000)]
        code = list(
            str(self.decimal_to_binary(large_param[0])).zfill(20)
            + str(self.decimal_to_binary(large_param[1])).zfill(20)
        )
        return code

    def change_to_param(self, code):
        large_c1 = int("".join(code[0:20]))
        large_c2 = int("".join(code[20:40]))
        param = [
            self.binary_to_decimal(large_c1) / 1000000,
            self.binary_to_decimal(large_c2) / 1000000,
        ]
        return param

    def get_Scores(self, Params):
        Scores = []
        for i in range(len(Params)):
            Scores.append(Params[i].score)
        return Scores

    def _sort(self, Params):
        Scores = self.get_Scores(Params)
        sorted_Scores_index = sorted(
            range(len(Scores)), key=lambda k: Scores[k], reverse=False
        )
        sorted_Params = []
        sorted_Scores = []
        for i in range(len(Params)):
            sorted_Params.append(Params[sorted_Scores_index[i]])
            sorted_Scores.append(Scores[sorted_Scores_index[i]])

        return sorted_Params, sorted_Scores

    def _delete(self, Params):
        new_Params = []
        for i in range(len(Params)):
            if (Params[i].p[0] <= (self.limit1 / 1000000)) & (
                Params[i].p[1] <= (self.limit2 / 1000000)
            ):
                new_Params.append(Params[i])
        return new_Params

    def roulette(self, Params):
        cumulation = []
        cumulation.append(
            0
        )  # 先補一個0，等等寫 if  r >= cumulation[j] and r <= cumulation[j+1]: 那裡比較方便
        mating_pool = []
        set_Params = list(set(Params))  # 雖然不太可能，但還是排除掉有重複元素的可能性
        Scores = self.get_Scores(set_Params)
        Scores_reci = []
        for k in range(len(Scores)):
            if Scores[k] == 0:
                Scores_reci.append(1)
            else:
                Scores_reci.append(1 / Scores[k])

        s = sum(Scores_reci)
        c = 0
        for i in range(len(Scores_reci)):
            c += Scores_reci[i] / s
            cumulation.append(c)
        for i in range(self.mating_pool_num):
            r = rd.random()
            for j in range(len(cumulation)):
                if r >= cumulation[j] and r <= cumulation[j + 1]:
                    mating_pool.append(set_Params[j])
                else:
                    pass

        return mating_pool

    def _crossover_and_mutate(self, mating_pool):
        print("Creating mating pool")
        next_mating_pool = []
        # crossover
        for _ in range(self.crossover_times):
            child1 = []
            child2 = []
            father, mother = rd.choices(mating_pool, k=2)
            father_list = self.change_to_code(father.p)
            mother_list = self.change_to_code(mother.p)

            index_start = rd.randrange(0, len(father_list) - self.crossover_length + 1)
            child1[:index_start] = father_list[0:index_start]
            child1[index_start : index_start + self.crossover_length] = mother_list[
                index_start : index_start + self.crossover_length
            ]
            child1[index_start + self.crossover_length :] = father_list[
                index_start + self.crossover_length :
            ]
            child2[:index_start] = mother_list[0:index_start]
            child2[index_start : index_start + self.crossover_length] = father_list[
                index_start : index_start + self.crossover_length
            ]
            child2[index_start + self.crossover_length :] = mother_list[
                index_start + self.crossover_length :
            ]

            # mutate
            r = rd.random()
            if r <= self.mutate:
                # print("mutate occurs !")
                idx = rd.randrange(0, len(child1) - 1)
                if child1[idx] == "1":
                    child1[idx] = "0"
                else:
                    child1[idx] = "1"

            child1_param = param(
                self.change_to_param(child1), self.iter_setting, self.true_ans
            )
            child2_param = param(
                self.change_to_param(child2), self.iter_setting, self.true_ans
            )

            next_mating_pool.append(child1_param)
            next_mating_pool.append(child2_param)
            # print("next_mating_pool = ", next_mating_pool)
        print("Finishing Crossover and Mutation")
        return next_mating_pool

    def evolution(self):
        extinct_count = 0
        Params = self._init_Params()  # 初始化亂數的Params

        record_dic = {
            "Level": [],
            "c1": [],
            "c2": [],
            "AvgDegree": [],
            "#Nodes_G": [],
            "Score": [],
        }
        Record = pd.DataFrame(record_dic)
        Record["Level"] = list(range(self.level))

        for i in range(self.level):  # 做level次
            print(f"The #{i} level")
            mating_pool = self.roulette(Params)
            Params = self._crossover_and_mutate(mating_pool)
            Params = self._delete(Params)
            if len(Params) == 0:
                extinct_count += 1
                print(f"Extinction {extinct_count}")
                Params = self._init_Params()  # 初始化亂數的Params
                if extinct_count >= 5:
                    print("Too many extinction")
                    return 0
                continue
            if extinct_count >= 5:
                print("Too many extinction")
                return 0
            gParams, gScores = self._sort(Params)  # g代表good，就是最後的解答，而且是sort過後的
            # best_param = gParams[0] # 用score排序後，第一個就是最好的答案
            # best_score = gScores[0]
            best10_param = gParams[0:10]
            best10_score = gScores[0:10]
            Record.iloc[i, 1] = best10_param[0].p[0]
            Record.iloc[i, 2] = best10_param[0].p[1]
            Record.iloc[i, 3] = best10_param[0].ans[0]
            Record.iloc[i, 4] = best10_param[0].ans[1]
            Record.iloc[i, 5] = best10_score[0]

        return best10_param, best10_score, Record


if __name__ == "__main__":
    pass
