import numpy as np

# returns 0 or 1
def kNN(X_train, y_train, X_test, max_k):
    assert (X_train.shape[0] == y_train.shape[0]), "numbers of training data and labels not match"
    assert (X_train.shape[1] == X_test.shape[1]), "dimensions of training data and testing data not match"
    assert (max_k>0), "K must greater than 0"

    result = np.zeros((max_k, X_test.shape[0]))

    for i in range(X_test.shape[0]):
        # calculate the distance to each training data
        distance = np.sum(np.absolute(X_train-X_test[i]), axis=1)
        # find the k nearest neighbors
        k_neighbors = np.argsort(distance)[:max_k]
        comp = [0, 0]
        for j in range(max_k):
            comp[int(y_train[int(k_neighbors[j])])] += 1
            result[j, i] = np.argmax(comp)
    return result