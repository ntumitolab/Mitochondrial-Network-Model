###############################################################################################
### 6X code
# import networkmodel
# import myGA
# import KLDivergence

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import multiprocessing
# import time

# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
# from scipy.stats import entropy
# from geneticalgorithm import geneticalgorithm as ga

# def get_ans(param):
#     netinfo = networkmodel.Iterate(param, 194, 400)
#     return [netinfo.iloc[-1,5], (netinfo.iloc[-1,12] / netinfo.iloc[-1,6]), (netinfo.iloc[-1,18] / netinfo.iloc[-1,6])]

# def get_answers(params):
#     with multiprocessing.Pool() as pool:
#         pool_out = pool.map(get_ans, params)
#         return pool_out

# def f(X):
#     num_of_sample = 30
#     params = [X] *num_of_sample
#     std_fail_count = 0
#     std_success_flag = 0
#     while (std_success_flag == 0) & (std_fail_count <= 20):
#         std_fail_count += 1
#         if std_fail_count >= 2:
#             print(std_fail_count)
#             print(np.array([x.std(), y.std(), z.std()]))
#         ANS = get_answers(params)
#         x = np.array([ans[0] for ans in ANS])
#         y = np.array([ans[1] for ans in ANS])
#         z = np.array([ans[2] for ans in ANS])
#         if 0 not in np.array([x.std(), y.std(), z.std()]):
#             std_success_flag = 1
#     dens1 = KLDivergence.KDE1V(x, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2 = KLDivergence.KDE1V(y, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3 = KLDivergence.KDE1V(z, variable_name='Ng2/N', bw_type='silverman', plot='F')
#     global x_ans
#     global y_ans
#     global z_ans
#     dens1_ans = KLDivergence.KDE1V(x_ans, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2_ans = KLDivergence.KDE1V(y_ans, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3_ans = KLDivergence.KDE1V(z_ans, variable_name='Ng2/N', bw_type='silverman', plot='F')

#     entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
#     entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
#     entro3 = KLDivergence.KLD1V(dens3, dens3_ans)

#     entro_final = ( entro1*0.45 + entro2*0.45 + entro3*0.1)
#     return entro_final

# if __name__ == '__main__':
#     start_time = time.time()

#     # create true answer from image analysis
#     fit_6 = pd.read_csv('./2D_glucose/6X_fitting.csv')
#     x_ans = np.array(fit_6['AvgDeg'])
#     y_ans = np.array(fit_6['Ng1/N'])
#     z_ans = np.array(fit_6['Ng2/N'])

#     """ create true answer from network model
#     true_params = [[0.03, 0.0015]] * 100
#     TRUE_ANS = get_answers(true_params)
#     x1_ans = np.array([ans[0] for ans in TRUE_ANS])
#     x2_ans = np.array([ans[1] for ans in TRUE_ANS])
#     x3_ans = np.array([ans[2] for ans in TRUE_ANS])

#     """
#     varbound = np.array([[0, 0.1], [0, 0.01]])
#     algorithm_param = {'max_num_iteration': 20,\
#                    'population_size':100,\
#                    'mutation_probability':0.6,\
#                    'elit_ratio': 0.03,\
#                    'crossover_probability': 0.7,\
#                    'parents_portion': 0.2,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':5}
#     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param,  convergence_curve=False)

#     model.run()
#     solution=model.output_dict
#     print(solution)
#     duration = time.time() - start_time
#     print(f"Duration {duration} seconds")

#     with open('./2D_glucose/output_6X.txt', 'a') as outFile:
#         outFile.write('\n' + '=========' + '\n')
#         outFile.write('6X' + '\n')
#         for mkey, mvalue in model.output_dict.items():
#             outFile.write(str(mkey)+'\n')
#             outFile.write(str(mvalue)+'\n')
        
#         outFile.write("Convergence :" + '\n')
#         for y in model.report:
#             outFile.write(str(y) + '\n')
#         outFile.write(f"Duration {duration} seconds")

###############################################################################################
### 3X code
# import networkmodel
# import myGA
# import KLDivergence

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import multiprocessing
# import time

# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
# from scipy.stats import entropy
# from geneticalgorithm import geneticalgorithm as ga

# def get_ans(param):
#     netinfo = networkmodel.Iterate(param, 176, 400)
#     return [netinfo.iloc[-1,5], (netinfo.iloc[-1,12] / netinfo.iloc[-1,6]), (netinfo.iloc[-1,18] / netinfo.iloc[-1,6])]

# def get_answers(params):
#     with multiprocessing.Pool() as pool:
#         pool_out = pool.map(get_ans, params)
#         return pool_out

# def f(X):
#     num_of_sample = 30
#     params = [X] *num_of_sample
#     std_fail_count = 0
#     std_success_flag = 0
#     while (std_success_flag == 0) & (std_fail_count <= 20):
#         std_fail_count += 1
#         if std_fail_count >= 2:
#             print(std_fail_count)
#             print(np.array([x.std(), y.std(), z.std()]))
#         ANS = get_answers(params)
#         x = np.array([ans[0] for ans in ANS])
#         y = np.array([ans[1] for ans in ANS])
#         z = np.array([ans[2] for ans in ANS])
#         if 0 not in np.array([x.std(), y.std(), z.std()]):
#             std_success_flag = 1
#     dens1 = KLDivergence.KDE1V(x, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2 = KLDivergence.KDE1V(y, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3 = KLDivergence.KDE1V(z, variable_name='Ng2/N', bw_type='silverman', plot='F')
#     global x_ans
#     global y_ans
#     global z_ans
#     dens1_ans = KLDivergence.KDE1V(x_ans, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2_ans = KLDivergence.KDE1V(y_ans, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3_ans = KLDivergence.KDE1V(z_ans, variable_name='Ng2/N', bw_type='silverman', plot='F')

#     entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
#     entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
#     entro3 = KLDivergence.KLD1V(dens3, dens3_ans)

#     entro_final = ( entro1*0.45 + entro2*0.45 + entro3*0.1)
#     return entro_final

# if __name__ == '__main__':
#     start_time = time.time()

#     # create true answer from image analysis
#     fit_3 = pd.read_csv('./2D_glucose/3X_fitting.csv')
#     x_ans = np.array(fit_3['AvgDeg'])
#     y_ans = np.array(fit_3['Ng1/N'])
#     z_ans = np.array(fit_3['Ng2/N'])

#     """ create true answer from network model
#     true_params = [[0.03, 0.0015]] * 100
#     TRUE_ANS = get_answers(true_params)
#     x1_ans = np.array([ans[0] for ans in TRUE_ANS])
#     x2_ans = np.array([ans[1] for ans in TRUE_ANS])
#     x3_ans = np.array([ans[2] for ans in TRUE_ANS])

#     """
#     varbound = np.array([[0, 0.1], [0, 0.01]])
#     algorithm_param = {'max_num_iteration': 20,\
#                    'population_size':100,\
#                    'mutation_probability':0.6,\
#                    'elit_ratio': 0.03,\
#                    'crossover_probability': 0.7,\
#                    'parents_portion': 0.2,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':5}
#     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param,  convergence_curve=False)

#     model.run()
#     solution=model.output_dict
#     print(solution)
#     duration = time.time() - start_time
#     print(f"Duration {duration} seconds")

#     with open('./2D_glucose/output_3X.txt', 'a') as outFile:
#         outFile.write('\n' + '=========' + '\n')
#         outFile.write('3X' + '\n')
#         for mkey, mvalue in model.output_dict.items():
#             outFile.write(str(mkey)+'\n')
#             outFile.write(str(mvalue)+'\n')
        
#         outFile.write("Convergence :" + '\n')
#         for y in model.report:
#             outFile.write(str(y) + '\n')
#         outFile.write(f"Duration {duration} seconds")


###############################################################################################
### 1X code
# import networkmodel
# import myGA
# import KLDivergence

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import multiprocessing
# import time

# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
# from scipy.stats import entropy
# from geneticalgorithm import geneticalgorithm as ga

# def get_ans(param):
#     netinfo = networkmodel.Iterate(param, 176, 400)
#     return [netinfo.iloc[-1,5], (netinfo.iloc[-1,12] / netinfo.iloc[-1,6]), (netinfo.iloc[-1,18] / netinfo.iloc[-1,6])]

# def get_answers(params):
#     with multiprocessing.Pool() as pool:
#         pool_out = pool.map(get_ans, params)
#         return pool_out

# def f(X):
#     num_of_sample = 30
#     params = [X] *num_of_sample
#     std_fail_count = 0
#     std_success_flag = 0
#     while (std_success_flag == 0) & (std_fail_count <= 20):
#         std_fail_count += 1
#         if std_fail_count >= 2:
#             print(std_fail_count)
#             print(np.array([x.std(), y.std(), z.std()]))
#         ANS = get_answers(params)
#         x = np.array([ans[0] for ans in ANS])
#         y = np.array([ans[1] for ans in ANS])
#         z = np.array([ans[2] for ans in ANS])
#         if 0 not in np.array([x.std(), y.std(), z.std()]):
#             std_success_flag = 1
#     dens1 = KLDivergence.KDE1V(x, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2 = KLDivergence.KDE1V(y, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3 = KLDivergence.KDE1V(z, variable_name='Ng2/N', bw_type='silverman', plot='F')
#     global x_ans
#     global y_ans
#     global z_ans
#     dens1_ans = KLDivergence.KDE1V(x_ans, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2_ans = KLDivergence.KDE1V(y_ans, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3_ans = KLDivergence.KDE1V(z_ans, variable_name='Ng2/N', bw_type='silverman', plot='F')

#     entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
#     entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
#     entro3 = KLDivergence.KLD1V(dens3, dens3_ans)

#     entro_final = ( entro1*0.45 + entro2*0.45 + entro3*0.1)
#     return entro_final

# if __name__ == '__main__':
#     start_time = time.time()

#     # create true answer from image analysis
#     fit_1 = pd.read_csv('./2D_glucose/1X_fitting.csv')
#     x_ans = np.array(fit_1['AvgDeg'])
#     y_ans = np.array(fit_1['Ng1/N'])
#     z_ans = np.array(fit_1['Ng2/N'])

#     """ create true answer from network model
#     true_params = [[0.03, 0.0015]] * 100
#     TRUE_ANS = get_answers(true_params)
#     x1_ans = np.array([ans[0] for ans in TRUE_ANS])
#     x2_ans = np.array([ans[1] for ans in TRUE_ANS])
#     x3_ans = np.array([ans[2] for ans in TRUE_ANS])

#     """
#     varbound = np.array([[0, 0.1], [0, 0.01]])
#     algorithm_param = {'max_num_iteration': 20,\
#                    'population_size':100,\
#                    'mutation_probability':0.6,\
#                    'elit_ratio': 0.03,\
#                    'crossover_probability': 0.7,\
#                    'parents_portion': 0.2,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':5}
#     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param,  convergence_curve=False)

#     model.run()
#     solution=model.output_dict
#     print(solution)
#     duration = time.time() - start_time
#     print(f"Duration {duration} seconds")

#     with open('./2D_glucose/output_1X.txt', 'a') as outFile:
#         outFile.write('\n' + '=========' + '\n')
#         outFile.write('1X' + '\n')
#         for mkey, mvalue in model.output_dict.items():
#             outFile.write(str(mkey)+'\n')
#             outFile.write(str(mvalue)+'\n')
        
#         outFile.write("Convergence :" + '\n')
#         for y in model.report:
#             outFile.write(str(y) + '\n')
#         outFile.write(f"Duration {duration} seconds")


###############################################################################################
### 0X code
import networkmodel
import myGA
import KLDivergence

import numpy as np
import pandas as pd
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
    netinfo = networkmodel.Iterate(param, 140, 400)
    return [netinfo.iloc[-1,5], (netinfo.iloc[-1,12] / netinfo.iloc[-1,6]), (netinfo.iloc[-1,18] / netinfo.iloc[-1,6])]

def get_answers(params):
    with multiprocessing.Pool() as pool:
        pool_out = pool.map(get_ans, params)
        return pool_out

def f(X):
    num_of_sample = 30
    params = [X] *num_of_sample
    std_fail_count = 0
    std_success_flag = 0
    while (std_success_flag == 0) & (std_fail_count <= 20):
        std_fail_count += 1
        if std_fail_count >= 2:
            print(std_fail_count)
            print(np.array([x.std(), y.std(), z.std()]))
        ANS = get_answers(params)
        x = np.array([ans[0] for ans in ANS])
        y = np.array([ans[1] for ans in ANS])
        z = np.array([ans[2] for ans in ANS])
        if 0 not in np.array([x.std(), y.std(), z.std()]):
            std_success_flag = 1
    dens1 = KLDivergence.KDE1V(x, variable_name='AvgDeg', bw_type='silverman', plot='F')
    dens2 = KLDivergence.KDE1V(y, variable_name='Ng1/N', bw_type='silverman', plot='F')
    dens3 = KLDivergence.KDE1V(z, variable_name='Ng2/N', bw_type='silverman', plot='F')
    global x_ans
    global y_ans
    global z_ans
    dens1_ans = KLDivergence.KDE1V(x_ans, variable_name='AvgDeg', bw_type='silverman', plot='F')
    dens2_ans = KLDivergence.KDE1V(y_ans, variable_name='Ng1/N', bw_type='silverman', plot='F')
    dens3_ans = KLDivergence.KDE1V(z_ans, variable_name='Ng2/N', bw_type='silverman', plot='F')

    entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
    entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
    entro3 = KLDivergence.KLD1V(dens3, dens3_ans)

    entro_final = ( entro1*0.45 + entro2*0.45 + entro3*0.1)
    return entro_final

if __name__ == '__main__':
    start_time = time.time()

    # create true answer from image analysis
    fit_0 = pd.read_csv('./2D_glucose/0X_fitting.csv')
    x_ans = np.array(fit_0['AvgDeg'])
    y_ans = np.array(fit_0['Ng1/N'])
    z_ans = np.array(fit_0['Ng2/N'])

    """ create true answer from network model
    true_params = [[0.03, 0.0015]] * 100
    TRUE_ANS = get_answers(true_params)
    x1_ans = np.array([ans[0] for ans in TRUE_ANS])
    x2_ans = np.array([ans[1] for ans in TRUE_ANS])
    x3_ans = np.array([ans[2] for ans in TRUE_ANS])

    """
    varbound = np.array([[0, 0.1], [0, 0.01]])
    algorithm_param = {'max_num_iteration': 20,\
                   'population_size':100,\
                   'mutation_probability':0.6,\
                   'elit_ratio': 0.03,\
                   'crossover_probability': 0.7,\
                   'parents_portion': 0.2,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':5}
    model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param,  convergence_curve=False)

    model.run()
    solution=model.output_dict
    print(solution)
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")

    with open('./2D_glucose/output_0X.txt', 'a') as outFile:
        outFile.write('\n' + '=========' + '\n')
        outFile.write('0X' + '\n')
        for mkey, mvalue in model.output_dict.items():
            outFile.write(str(mkey)+'\n')
            outFile.write(str(mvalue)+'\n')
        
        outFile.write("Convergence :" + '\n')
        for y in model.report:
            outFile.write(str(y) + '\n')
        outFile.write(f"Duration {duration} seconds")









##############################################################################################
## FCCP code
# import networkmodel
# import myGA
# import KLDivergence

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import multiprocessing
# import time

# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut
# from scipy.stats import entropy
# from geneticalgorithm import geneticalgorithm as ga

# def get_ans(param):
#     netinfo = networkmodel.Iterate(param, 140, 400)
#     return [netinfo.iloc[-1,5], (netinfo.iloc[-1,12] / netinfo.iloc[-1,6]), (netinfo.iloc[-1,18] / netinfo.iloc[-1,6])]

# def get_answers(params):
#     with multiprocessing.Pool() as pool:
#         pool_out = pool.map(get_ans, params)
#         return pool_out

# def f(X):
#     num_of_sample = 30
#     params = [X] *num_of_sample
#     std_fail_count = 0
#     std_success_flag = 0
#     while (std_success_flag == 0) & (std_fail_count <= 20):
#         std_fail_count += 1
#         if std_fail_count >= 2:
#             print(std_fail_count)
#             print(np.array([x.std(), y.std(), z.std()]))
#         ANS = get_answers(params)
#         x = np.array([ans[0] for ans in ANS])
#         y = np.array([ans[1] for ans in ANS])
#         z = np.array([ans[2] for ans in ANS])
#         if 0 not in np.array([x.std(), y.std(), z.std()]):
#             std_success_flag = 1
#     dens1 = KLDivergence.KDE1V(x, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2 = KLDivergence.KDE1V(y, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3 = KLDivergence.KDE1V(z, variable_name='Ng2/N', bw_type='silverman', plot='F')
#     global x_ans
#     global y_ans
#     global z_ans
#     dens1_ans = KLDivergence.KDE1V(x_ans, variable_name='AvgDeg', bw_type='silverman', plot='F')
#     dens2_ans = KLDivergence.KDE1V(y_ans, variable_name='Ng1/N', bw_type='silverman', plot='F')
#     dens3_ans = KLDivergence.KDE1V(z_ans, variable_name='Ng2/N', bw_type='silverman', plot='F')

#     entro1 = KLDivergence.KLD1V(dens1, dens1_ans)
#     entro2 = KLDivergence.KLD1V(dens2, dens2_ans)
#     entro3 = KLDivergence.KLD1V(dens3, dens3_ans)
#     # with open('output_test.txt', 'a') as outFile:
#     #     outFile.write('\n' + '=========' + '\n')
#     #     outFile.write('y' + '\n')
#     #     outFile.write(str(y)+ '\n')
#     #     outFile.write('y_ans' + '\n')
#     #     outFile.write(str(y_ans)+ '\n')
#     #     outFile.write('dens2' + '\n')
#     #     outFile.write(str(dens2) + '\n')
#     #     outFile.write('dens2_ans' + '\n')
#     #     outFile.write(str(dens2_ans) + '\n')
#     #     outFile.write('entro' + '\n')
#     #     outFile.write(str(entro1) + '\n')
#     #     outFile.write(str(entro2) + '\n')
#     #     outFile.write(str(entro3) + '\n')
 
#     # print(dens2)
#     # print('=====')
#     # print(dens2_ans)
#     # print('=====')
#     # print('=====')
#     # print(y)
#     # print(y_ans)
#     # print(f'entro1 = {entro1}')
#     # print(f'entro2 = {entro2}')
#     # print(f'entro3 = {entro3}')
#     entro_final = ( entro1*0.7 + entro2*0.15 + entro3*0.15 )
#     return entro_final

# if __name__ == '__main__':
#     start_time = time.time()

#     # create true answer from image analysis
#     fit_f = pd.read_csv('FCCP_fitting.csv')
#     x_ans = np.array(fit_f['AvgDeg'])
#     y_ans = np.array(fit_f['Ng1/N'])
#     z_ans = np.array(fit_f['Ng2/N'])

#     """ create true answer from network model
#     true_params = [[0.03, 0.0015]] * 100
#     TRUE_ANS = get_answers(true_params)
#     x1_ans = np.array([ans[0] for ans in TRUE_ANS])
#     x2_ans = np.array([ans[1] for ans in TRUE_ANS])
#     x3_ans = np.array([ans[2] for ans in TRUE_ANS])

#     """
#     varbound = np.array([[0, 0.1], [0, 0.01]])
#     algorithm_param = {'max_num_iteration': 20,\
#                    'population_size':100,\
#                    'mutation_probability':0.6,\
#                    'elit_ratio': 0.03,\
#                    'crossover_probability': 0.7,\
#                    'parents_portion': 0.2,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':5}
#     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param,  convergence_curve=False)

#     model.run()
#     solution=model.output_dict
#     print(solution)
#     duration = time.time() - start_time
#     print(f"Duration {duration} seconds")

#     with open('output.txt', 'a') as outFile:
#         outFile.write('\n' + '=========' + '\n')
#         outFile.write('FCCP' + '\n')
#         for mkey, mvalue in model.output_dict.items():
#             outFile.write(str(mkey)+'\n')
#             outFile.write(str(mvalue)+'\n')
        
#         outFile.write("Convergence :" + '\n')
#         for y in model.report:
#             outFile.write(str(y) + '\n')
#         outFile.write(f"Duration {duration} seconds")

# ## 2 varaibles version
# # def f(X):
# #     num_of_sample = 100
# #     params = [X] *num_of_sample
# #     ANS = get_answers(params)
# #     x = np.array([ans[0] for ans in ANS])
# #     y = np.array([ans[1] for ans in ANS])
# #     dens = KLDivergence.KDE2V(x, y, 'silverman', plot='F')
# #     global x_ans
# #     global y_ans
# #     dens_ans = KLDivergence.KDE2V(x_ans, y_ans, 'silverman', plot='F')
# #     entro = KLDivergence.KLD2V(dens, dens_ans)
# #     return entro

# # if __name__ == '__main__':
# #     start_time = time.time()

# #     true_params = [[0.03, 0.0015]] * 100
# #     TRUE_ANS = get_answers(true_params)
# #     x_ans = np.array([ans[0] for ans in TRUE_ANS])
# #     y_ans = np.array([ans[1] for ans in TRUE_ANS])

# #     varbound = np.array([[0, 0.1], [0, 0.01]])
# #     algorithm_param = {'max_num_iteration': 30,\
# #                    'population_size':100,\
# #                    'mutation_probability':0.6,\
# #                    'elit_ratio': 0.03,\
# #                    'crossover_probability': 0.7,\
# #                    'parents_portion': 0.2,\
# #                    'crossover_type':'uniform',\
# #                    'max_iteration_without_improv':8}
# #     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param)

# #     model.run()
# #     solution=model.output_dict
# #     print(solution)
# #     duration = time.time() - start_time
# #     print(f"Duration {duration} seconds")


# ## 3 varaibles version
# # def f(X):
# #     num_of_sample = 30
# #     params = [X] *num_of_sample
# #     ANS = get_answers(params)
# #     x = np.array([ans[0] for ans in ANS])
# #     y = np.array([ans[1] for ans in ANS])
# #     z = np.array([ans[2] for ans in ANS])
# #     dens = KLDivergence.KDE3V(x, y, z, 'silverman', plot='F')
# #     global x_ans
# #     global y_ans
# #     global z_ans
# #     dens_ans = KLDivergence.KDE3V(x_ans, y_ans, z_ans, 'silverman', plot='F')
# #     entro = KLDivergence.KLD3V(dens, dens_ans)
# #     return entro

# # if __name__ == '__main__':
# #     start_time = time.time()

# #     # fit_0X = pd.read_csv('0X_fitting.csv')
# #     # x_ans = np.array(fit_0X['newAvgDeg'])
# #     # y_ans = np.array(fit_0X['Ng1/N'])
# #     # z_ans = np.array(fit_0X['Ng2/N'])

# #     fit_c = pd.read_csv('control_fitting.csv')
# #     x_ans = np.array(fit_c['AvgDeg'])
# #     y_ans = np.array(fit_c['Ng1/N'])
# #     z_ans = np.array(fit_c['Ng2/N'])

# #     # create true answer from network model
# #     # true_params = [[0.03, 0.0015]] * 100
# #     # TRUE_ANS = get_answers(true_params)
# #     # x_ans = np.array([ans[0] for ans in TRUE_ANS])
# #     # y_ans = np.array([ans[1] for ans in TRUE_ANS])
# #     # z_ans = np.array([ans[2] for ans in TRUE_ANS])

# #     varbound = np.array([[0, 0.1], [0, 0.01]])
# #     algorithm_param = {'max_num_iteration': 20,\
# #                    'population_size':100,\
# #                    'mutation_probability':0.6,\
# #                    'elit_ratio': 0.03,\
# #                    'crossover_probability': 0.7,\
# #                    'parents_portion': 0.2,\
# #                    'crossover_type':'uniform',\
# #                    'max_iteration_without_improv':5}
# #     model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound, function_timeout=600, algorithm_parameters=algorithm_param, convergence_curve=False)

# #     model.run()
# #     solution=model.output_dict
# #     print(solution)
# #     duration = time.time() - start_time

# #     print(f"Duration {duration} seconds")

# #     with open('output.txt', 'a') as outFile:
# #         outFile.write('\n' + '=========' + '\n')
# #         outFile.write('control' + '\n')
# #         for mkey, mvalue in model.output_dict.items():
# #             outFile.write(str(mkey)+'\n')
# #             outFile.write(str(mvalue)+'\n')
        
# #         outFile.write("Convergence :" + '\n')
# #         for y in model.report:
# #             outFile.write(str(y) + '\n')
# #         outFile.write(f"Duration {duration} seconds")
    



# # true_param = [0.03, 0.0015]
# # num_of_sample = 100
# # TRUE_ANS = []
# # start_time = time.time()
# # for i in range(num_of_sample):
# #     netinfo_ans = networkmodel.Iterate(true_param, 200, 200)
# #     TRUE_ANS.append([netinfo_ans.iloc[-1,5], netinfo_ans.iloc[-1,7]])
# # print(TRUE_ANS)
# # duration = time.time() - start_time
# # print(f"Duratioin {duration} seconds")
# # x_ans = np.array([ans[0] for ans in TRUE_ANS])
# # y_ans = np.array([ans[1] for ans in TRUE_ANS])
# # plt.scatter(x_ans, y_ans, c='black', s=20, edgecolor='white')
# # plt.show()