import numpy as np
import matplotlib.pyplot as plt

# 1st step: set_k()
# 2nd step: train()
# result stored in .obj
class k_means(object):
    
    def __init__(self, data_train):
        self.data_train = data_train

    def set_k(self, k):
        self.k = k
        self.reset_c()

    def reset_c(self):
        self.c = np.random.rand(self.k, 2)
        self.obj = []

    def update_c(self):
        for i in range(self.k):
            self.c[i, :] = np.mean(self.data_train[self.result[:,0] == i],
                                    axis=0)

    def min_dis(self, point):
        errors = np.sum((self.c - point)**2, axis=1)
        return (np.argmin(errors),errors[np.argmin(errors)])

    def update_u(self):
        self.result = np.apply_along_axis(self.min_dis,
                                            1, self.data_train)
        self.obj.append(np.sum(self.result[:,1]))

    def train(self, itr):
        for i in range(itr):
            self.update_u()
            self.update_c()

