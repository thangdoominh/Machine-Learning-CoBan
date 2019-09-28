from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

# create data, K cluster, N data, draw graph
means = [[2,2], [8,3], [3,6]]
conv = [[1, 0], [0,1]]
N = 500
K = 3

X0 = np.random.multivariate_normal(means[0], conv, N)
X1 = np.random.multivariate_normal(means[1], conv, N)
X2 = np.random.multivariate_normal(means[2], conv, N)

X = np.concatenate((X0, X1, X2), axis = 0)

original_labels = np.asarray([0]*N + [1]*N + [2]*N).T

# print(original_labels)
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]


    plt.plot(X0[:, 0], X0[:, 1], 'b^')
    plt.plot(X1[:, 0], X1[:, 1], 'go')
    plt.plot(X2[:, 0], X2[:, 1], 'rs')

    plt.axis('equal')
    plt.show()
    # print(K)

# Kmeans_display(X, original_labels)

# Khởi tạo điểm center ban đầu
def kmeans_choice_centers(X, K):
    return X[np.random.choice( X.shape[0], K,  replace = False)]

# labels for data
def kmeans_assign_labels(X, centers):
    D = cdist(X,centers)
    return np.argmin(D, axis = 1)

# update new_center, numpy.mean
def kmeans_update_center(X, label, K):

    new_centers = np.zeros((K, X.shape[1]))

    for k in range(K):

        Xk = X[label == k, :]

        new_centers[k, :] = np.mean(Xk, axis = 0)

    return new_centers

# check has converged
def has_converged( centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]) )
# update new_labels
# calculator

def kmean(X, K):
    centers = [kmeans_choice_centers(X,K)]
    labels = []
    count = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_center(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        count += 1
    return (centers, labels, count)
(centers, labels, count) = kmean(X, K)
print('Centers found by our algorithm: ')
print(centers[-1])
kmeans_display(X, labels[-1])
