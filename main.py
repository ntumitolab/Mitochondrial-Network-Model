##############################################################################################
## 0X code

import networkmodel
import myGA
import KLDivergence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import multiprocessing
import time
import os

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import entropy
from geneticalgorithm import geneticalgorithm as ga


def get_ans(param):
    netinfo = networkmodel.Iterate(param, 140, 400)
    # netinfo = networkmodel.Iterate(param, 4, 10)
    return [
        netinfo.iloc[-1, 5],
        (netinfo.iloc[-1, 12] / netinfo.iloc[-1, 6]),
        (netinfo.iloc[-1, 18] / netinfo.iloc[-1, 6]),
    ]


def get_answers(paramsets):
    with multiprocessing.Pool() as pool:
        pool_out = pool.map(get_ans, paramsets)
        return pool_out


def f(X, x_ans, y_ans, z_ans, num_of_samples=30, w1=0.45, w2=0.45, w3=0.1):
    parassets = [X] * num_of_samples
    std_fail_count = 0
    std_success_flag = False
    while (not std_success_flag) and (std_fail_count <= 20):
        std_fail_count += 1
        if std_fail_count >= 2:
            print(std_fail_count)
            print(np.array([x.std(), y.std(), z.std()]))
        ANS = get_answers(parassets)
        x = np.array([ans[0] for ans in ANS])
        y = np.array([ans[1] for ans in ANS])
        z = np.array([ans[2] for ans in ANS])
        std_success_flag = all((x.std(), y.std(), z.std()))

    dens1 = KLDivergence.KDE1V(x, variable_name="AvgDeg", bw_type="silverman", plot="F")
    dens2 = KLDivergence.KDE1V(y, variable_name="Ng1/N", bw_type="silverman", plot="F")
    dens3 = KLDivergence.KDE1V(z, variable_name="Ng2/N", bw_type="silverman", plot="F")
    dens1_ans = KLDivergence.KDE1V(
        x_ans, variable_name="AvgDeg", bw_type="silverman", plot="F"
    )
    dens2_ans = KLDivergence.KDE1V(
        y_ans, variable_name="Ng1/N", bw_type="silverman", plot="F"
    )
    dens3_ans = KLDivergence.KDE1V(
        z_ans, variable_name="Ng2/N", bw_type="silverman", plot="F"
    )

    entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
    entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
    entro3 = KLDivergence.KLD1V(dens3, dens3_ans)

    entro_weighted = entro1 * w1 + entro2 * w2 + entro3 * w3
    return entro_weighted


def run(
    datapath,
    w1=0.45,
    w2=0.45,
    w3=0.1,
    ga_params={
        "max_num_iteration": 20,
        "population_size": 100,
        "mutation_probability": 0.6,
        "elit_ratio": 0.03,
        "crossover_probability": 0.7,
        "parents_portion": 0.2,
        "crossover_type": "uniform",
        "max_iteration_without_improv": 5,
    },
):
    start_time = time.time()
    # create true answer from image analysis
    fit_0 = pd.read_csv(datapath)
    x_ans = np.array(fit_0["AvgDeg"])
    y_ans = np.array(fit_0["Ng1/N"])
    z_ans = np.array(fit_0["Ng2/N"])
    varbound = np.array([[0, 0.1], [0, 0.01]])
    model = ga(
        function=lambda X: f(X, x_ans, y_ans, z_ans, w1=w1, w2=w2, w3=w3),
        dimension=2,
        variable_type="real",
        variable_boundaries=varbound,
        function_timeout=600,
        algorithm_parameters=ga_params,
        convergence_curve=False,
    )
    model.run()

    solution = model.output_dict
    print("Solution:", solution)
    duration = time.time() - start_time
    print(f"Took: {duration} seconds")

    outfilepath = os.path.splitext(datapath)[0] + ".txt"

    with open(outfilepath, "a") as outFile:
        outFile.write("Source: " + datapath)
        outFile.write("\n" + "=========" + "\n")
        for mkey, mvalue in model.output_dict.items():
            outFile.write(str(mkey) + "\n")
            outFile.write(str(mvalue) + "\n")

        outFile.write("Convergence :" + "\n")
        for y in model.report:
            outFile.write(str(y) + "\n")
        outFile.write(f"Duration {duration} seconds")


if __name__ == "__main__":

    # create true answer from image analysis
    run("data/2D_glucose/0X_fitting.csv")
    run("data/2D_glucose/1X_fitting.csv")
    run("data/2D_glucose/3X_fitting.csv")
    run("data/2D_glucose/6X_fitting.csv")

    gaargs = {
        "max_num_iteration": 20,
        "population_size": 20,
        "mutation_probability": 0.6,
        "elit_ratio": 0.03,
        "crossover_probability": 0.7,
        "parents_portion": 0.2,
        "crossover_type": "uniform",
        "max_iteration_without_improv": 5,
    }
    w1 = w2 = w3 = 1 / 3

    run("data/Oligomycin_fitting.csv", ga_params=gaargs, w1=w1, w2=w2, w3=w3)
    run("data/control_fitting.csv", ga_params=gaargs, w1=w1, w2=w2, w3=w3)
    run("data/FCCP_fitting.csv", ga_params=gaargs, w1=w1, w2=w2, w3=w3)
    run("data/Rotenone_fitting.csv", ga_params=gaargs, w1=w1, w2=w2, w3=w3)
