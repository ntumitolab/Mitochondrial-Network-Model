import networkmodel
import myGA
import KLDivergence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import multiprocessing
import time

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import entropy
from geneticalgorithm import geneticalgorithm as ga


def get_ans(param):
    netinfo = networkmodel.Iterate(param, 200, 200)
    return [netinfo.iloc[-1,5], netinfo.iloc[-1,7]]

def get_answers(params):
    with multiprocessing.Pool() as pool:
        pool_out = pool.map(get_ans, params)
        return pool_out
if __name__ == '__main__':
    params = [[0.03, 0.0015]] * 100
    start_time = time.time()
    ANS = get_answers(params)
    print(ANS)
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")




# true_param = [0.03, 0.0015]
# num_of_sample = 100
# TRUE_ANS = []
# start_time = time.time()
# for i in range(num_of_sample):
#     netinfo_ans = networkmodel.Iterate(true_param, 200, 200)
#     TRUE_ANS.append([netinfo_ans.iloc[-1,5], netinfo_ans.iloc[-1,7]])
# print(TRUE_ANS)
# duration = time.time() - start_time
# print(f"Duratioin {duration} seconds")
# x_ans = np.array([ans[0] for ans in TRUE_ANS])
# y_ans = np.array([ans[1] for ans in TRUE_ANS])
# plt.scatter(x_ans, y_ans, c='black', s=20, edgecolor='white')
# plt.show()