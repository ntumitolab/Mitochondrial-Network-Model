import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import entropy
from scipy import stats

# https://stackoverflow.com/questions/63812970/scipy-gaussian-kde-matrix-is-not-positive-definite
# class GaussianKde(stats.gaussian_kde):
#         """
#         Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
#         to the covmat eigenvalues, to prevent exceptions due to numerical error.
#         """

#         EPSILON = 1e-10  # adjust this at will

#         def _compute_covariance(self):
#             """Computes the covariance matrix for each Gaussian kernel using
#             covariance_factor().
#             """

#             self.factor = self.covariance_factor()
#             # Cache covariance and inverse covariance of the data
#             if not hasattr(self, '_data_inv_cov'):
#                 self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
#                                                              bias=False,
#                                                              aweights=self.weights))
#                 # we're going the easy way here
#                 self._data_covariance += self.EPSILON * np.eye(
#                     len(self._data_covariance))
#                 self._data_inv_cov = np.linalg.inv(self._data_covariance)
#                 print('05')
#                 print(self._data_covariance)
#             self.covariance = self._data_covariance * self.factor**2
#             self.inv_cov = self._data_inv_cov / self.factor**2
#             L = np.linalg.cholesky(self.covariance * 2 * np.pi)
#             self._norm_factor = 2*np.log(np.diag(L)).sum()  # needed for scipy 1.5.2
#             self.log_det = 2*np.log(np.diag(L)).sum()  # changed var name on 1.6.2


def KDE3V(x, y, z, bw_type="grid", plot="T"):
    xyz = np.vstack([x, y, z])
    if bw_type == "grid":
        bandwidths = 10 ** np.linspace(-1, 1, 100)

        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": bandwidths},
            cv=LeaveOneOut(),
        )
        grid.fit(xyz.T)
        bw = grid.best_params_["bandwidth"]
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(xyz.T)

        xmin = 0
        xmax = 3
        ymin = 0
        ymax = 1
        zmin = 0
        zmax = 1

        X, Y, Z = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        gdens = np.exp(kde.score_samples(positions.T))

    elif bw_type == "silverman":
        xmin = 0
        xmax = 3
        ymin = 0
        ymax = 1
        zmin = 0
        zmax = 1
        X, Y, Z = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

        kde = stats.gaussian_kde(xyz)
        kde.set_bandwidth(bw_method="scott")
        gdens = kde(positions).T

    else:
        print("Wrong bw_type")

    return gdens


# def KDE3V(x, y, z, bw_type = 'grid', plot='T'):
#     xyz = np.vstack([x,y,z])
#     if bw_type == 'grid':
#         bandwidths = 10 ** np.linspace(-1, 1, 100)

#         grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                             {'bandwidth': bandwidths},
#                             cv=LeaveOneOut())
#         grid.fit(xyz.T)
#         bw = grid.best_params_['bandwidth']

#     elif bw_type == 'silverman':
#         d = xyz.shape[0]
#         n = xyz.shape[1]
#         bw = (n * (d + 2) / 4.)**(-1. / (d + 4))


#     else:
#         print('Wrong bw_type')

#     # instantiate and fit the KDE model
#     kde = KernelDensity(bandwidth=bw, kernel='gaussian')
#     kde.fit(xyz.T)

#     # xmin = x.min()
#     # xmax = x.max()
#     # ymin = y.min()
#     # ymax = y.max()
#     # zmin = z.min()
#     # zmax = z.max()
#     xmin = 0
#     xmax = 3
#     ymin = 0
#     ymax = 1
#     zmin = 0
#     zmax = 1

#     X, Y, Z = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
#     positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
#     gdens = np.exp(kde.score_samples(positions.T))

#     return gdens

# 2 variables KLD
def KLD3V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2, base=2)


def KDE2V(x, y, bw_type="grid", plot="T"):
    xy = np.vstack([x, y])
    if bw_type == "grid":
        bandwidths = 10 ** np.linspace(-1, 1, 100)

        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": bandwidths},
            cv=LeaveOneOut(),
        )
        grid.fit(xy.T)
        bw = grid.best_params_["bandwidth"]

    elif bw_type == "silverman":
        d = xy.shape[0]
        n = xy.shape[1]
        bw = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

    else:
        print("Wrong bw_type")

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
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

    if plot == "T":
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.imshow(
            np.rot90(Z), cmap=plt.get_cmap("viridis"), extent=[xmin, xmax, ymin, ymax]
        )

        ax.scatter(x, y, c="red", s=20, edgecolor="red")
        # ax.set_aspect('auto')
        plt.show()
    else:
        pass

    return gdens


# 2 variables KLD
def KLD2V(gdens1, gdens2):
    return entropy(pk=gdens1, qk=gdens2, base=2)


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
        print("=====")
        print(x.std())

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
    return entropy(pk=gdens1, qk=gdens2, base=2)
