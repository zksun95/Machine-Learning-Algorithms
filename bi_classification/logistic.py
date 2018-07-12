import numpy as np
from scipy.special import expit

# use Newton's method or steepest ascent algorithm
# returns 1 or -1
def logistic(X_train, y_train, X_test, iter, Newtons_methos=False):
    weights = np.zeros_like(X_train[0]).T
    l = []
    if(Newtons_method):
        for t in range(iter):
            delta = 1/np.sqrt(t+2)
            #sigmoid = y_train * X_train.dot(weights)
            #sigmoid = 1/(1+np.exp(-sigmoid))
            sigmoid = expit(np.multiply(y_train.reshape((-1, 1)), X_train.dot(weights)))
            # add a small number to avoid inf
            l.append(np.sum(np.log(sigmoid+1e-10)))
            theLongExpression = -np.multiply(
                                    np.multiply(sigmoid,1-sigmoid),
                                    X_train).T.dot(X_train)
            theShorterExpress = X_train.T.dot(np.multiply(y_train.reshape((-1, 1)), 1-sigmoid))
            step = np.linalg.inv(theLongExpression).dot(theShorterExpress)
            weights -= delta*step
    else:
        for t in range(iter):
            delta = 1/(1e5*np.sqrt(t+2))
            #sigmoid = y_train * X_train.dot(weights)
            #sigmoid = 1/(1+np.exp(-sigmoid))
            sigmoid = expit(y_train * X_train.dot(weights))
            # add a small number to avoid inf
            l.append(np.sum(np.log(sigmoid+1e-10)))
            step = X_train.T.dot(y_train * (1-sigmoid))
            weights += delta*step

    return np.sign(X_test.dot(weights))