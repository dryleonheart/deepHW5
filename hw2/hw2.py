'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    loss = 0.0;
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x=np.reshape(x.T, (-1, n))

    s = W@x+b

    exp_s = np.exp(s)

    sum_exp_s = np.sum(exp_s)

    soft_max = exp_s / sum_exp_s

    for i in range(n):
        temp = soft_max[y[i],i]
        loss -= np.log(temp)
    
    loss /= soft_max.shape[0]



    return loss

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss
    loss = 0.0;
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x=np.reshape(x.T, (-1, n))

    s = W@x+b

    for i in range(n):
        score = s[:,i]
        temp = score[y[i]]
        for j in range(num_class):
            if(j == y[i]):
                continue
            margin = score[j] - temp + 1
            if margin > 0:
                loss += margin

    loss /= n;

    return loss

########################################
# Part 3. kNN classification
########################################
def knn_test(x_train, y_train, x_test, y_test, k):
    # implement your function here
    #return accuracy
    print('x_train shape [0] , shape [1] = ' , x_train.shape[0] , ' , ' , x_train.shape[1])
    print('y_train : ',y_train)

    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    distance = np.zeros((n_test, n_train))

    Multi = np.dot(x_test, x_train.T)
    distance = np.sqrt(-2*Multi + np.matrix(np.square(x_train).sum(axis=1))+ np.matrix(np.square(x_test).sum(axis=1)).T)

    print('return distance = ' , distance )
    cnt = 0
    for i in range(n_test):
        score = distance[i,:]
        s = score.argsort()
        sortedy = y_train[s]
        chk=np.zeros(k)
        for j in range(k):
            chk[j] += sortedy[0][j]
        mode = stats.mode(chk)
        if(mode[0] == y_test[i]):
            cnt +=1

    return cnt/n_test


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]
    
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x@W.T + b
    # score has shape (n_test, num_class)
    
    # get argmax over class dim
    res = np.argmax(s, axis = 1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum()/n_test
    
    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'softmax'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, k)*100
          , '% for k value of ', k)
