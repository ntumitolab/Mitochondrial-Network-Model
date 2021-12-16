import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import entropy
from scipy import stats


def KDE1V(x, variable_name, bw_type="grid", plot="T"):
    if bw_type == "grid":
        bandwidths = 10 ** np.linspace(-1, 1, 100)

        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": bandwidths},
            cv=LeaveOneOut(),
        )
        grid.fit(x[:, None])
        bw = grid.best_params_["bandwidth"]
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(x[:, None])

        if variable_name == "AvgDeg":
            xmin = 0
            xmax = 3
        if variable_name == "Ng1/N":
            xmin = 0
            xmax = 1
        if variable_name == "Ng2/N":
            xmin = 0
            xmax = 1

        X = np.mgrid[xmin:xmax:100j]
        positions = np.vstack([X.ravel()])
        gdens = np.exp(kde.score_samples(positions.T))

    elif bw_type == "silverman":
        if variable_name == "AvgDeg":
            xmin = 0
            xmax = 3
        if variable_name == "Ng1/N":
            xmin = 0
            xmax = 1
        if variable_name == "Ng2/N":
            xmin = 0
            xmax = 1

        X = np.mgrid[xmin:xmax:100j]
        positions = np.vstack([X.ravel()])
        # print("=====")
        # print(x.std())

        kde = stats.gaussian_kde(x)

        kde.set_bandwidth(bw_method="silverman")
        gdens = kde(positions).T

    else:
        print("Wrong bw_type")

    # if plot == 'T':
    #     fig = plt.figure(figsize=(12,10))
    #     ax = fig.add_subplot(111)
    #     ax.imshow(np.rot90(Z), cmap=plt.get_cmap('viridis'),
    #               extent=[xmin, xmax, ymin, ymax])

    #     ax.scatter(x, y, c='red', s=20, edgecolor='red')
    #     #ax.set_aspect('auto')
    #     plt.show()
    # else:
    #     pass

    return gdens


# 1 variables KLD
def KLD1V(gdens1, gdens2):
    if (0 in gdens1) or (0 in gdens2):
        gdens1 = [gd + 1e-100 for gd in gdens1]
        gdens2 = [gd + 1e-100 for gd in gdens2]

    if entropy(pk=gdens1, qk=gdens2, base=2) >= entropy(pk=gdens2, qk=gdens1, base=2):
        return entropy(pk=gdens2, qk=gdens1, base=2)
    else:
        return entropy(pk=gdens1, qk=gdens2, base=2)


""" Old version of KLD1V
def KLD1V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2,base=2)

"""


"""
def KDE3V(x, y, z, bw_type = 'grid', plot='T'):
    xyz = np.vstack([x,y,z])
    if bw_type == 'grid':
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(xyz.T)
        bw = grid.best_params_['bandwidth']
        
    elif bw_type == 'silverman':
        d = xyz.shape[0]
        n = xyz.shape[1]
        bw = (n * (d + 2) / 4.)**(-1. / (d + 4))
        
        
    else:
        print('Wrong bw_type')
        
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(xyz.T)

    # xmin = x.min()
    # xmax = x.max()
    # ymin = y.min()
    # ymax = y.max()
    # zmin = z.min()
    # zmax = z.max()
    xmin = 0
    xmax = 3
    ymin = 0
    ymax = 1
    zmin = 0
    zmax = 1

    X, Y, Z = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    gdens = np.exp(kde.score_samples(positions.T))

    return gdens

# 2 variables KLD
def KLD3V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2,base=2)


def KDE2V(x, y, bw_type = 'grid', plot='T'):
    xy = np.vstack([x,y])
    if bw_type == 'grid':
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(xy.T)
        bw = grid.best_params_['bandwidth']
        
    elif bw_type == 'silverman':
        d = xy.shape[0]
        n = xy.shape[1]
        bw = (n * (d + 2) / 4.)**(-1. / (d + 4))
        
        
    else:
        print('Wrong bw_type')
        
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(xy.T)

    # xmin = x.min()
    # xmax = x.max()
    # ymin = y.min()
    # ymax = y.max()
    # xmin = 0
    # xmax = 3
    # ymin = 0
    # ymax = y.max()+5
    xmin = 0
    xmax = 3
    ymin = 0
    ymax = 1

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    gdens = np.exp(kde.score_samples(positions.T))
    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    if plot == 'T': 
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('viridis'),
                  extent=[xmin, xmax, ymin, ymax])

        ax.scatter(x, y, c='red', s=20, edgecolor='red')
        #ax.set_aspect('auto')
        plt.show()
    else:
        pass
    
    return gdens

# 2 variables KLD
def KLD2V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2,base=2)

def KDE1V(x, variable_name, bw_type = 'grid', plot='T'):
    if bw_type == 'grid':
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(x[:, None])
        bw = grid.best_params_['bandwidth']
        
    elif bw_type == 'silverman':
        d = x[:, None].shape[0]
        n = x[:, None].shape[1]
        bw = (n * (d + 2) / 4.)**(-1. / (d + 4))
        
        
    else:
        print('Wrong bw_type')
        
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(x[:, None])

    # xmin = x.min()
    # xmax = x.max()
    if variable_name == 'AvgDeg':
        xmin = 0
        xmax = 3
    if variable_name == 'Ng1/N':
        xmin = 0
        xmax = 1
    if variable_name == 'Ng2/N':
        xmin = 0
        xmax = 1  

    X= np.mgrid[xmin:xmax:100j]
    positions = np.vstack([X.ravel()])
    gdens = np.exp(kde.score_samples(positions.T))

    # if plot == 'T': 
    #     fig = plt.figure(figsize=(12,10))
    #     ax = fig.add_subplot(111)
    #     ax.imshow(np.rot90(Z), cmap=plt.get_cmap('viridis'),
    #               extent=[xmin, xmax, ymin, ymax])

    #     ax.scatter(x, y, c='red', s=20, edgecolor='red')
    #     #ax.set_aspect('auto')
    #     plt.show()
    # else:
    #     pass
    
    return gdens

# 1 variables KLD
def KLD1V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2,base=2)


"""
