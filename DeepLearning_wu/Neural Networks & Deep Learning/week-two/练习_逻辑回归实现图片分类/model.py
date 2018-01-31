import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset       # 加载数据的模块
def lr_model(num_iterations, learning_rate, print_cost):
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()    # 加载数据，classes代表类别
    train_set_x,test_set_x=dealimg(train_set_x_orig,test_set_x_orig)        # 向量化、归一化
    X_train, Y_train, X_test, Y_test=train_set_x,train_set_y,test_set_x,test_set_y
    w, b = initialize_with_zeros(X_train.shape[0])      # initialize parameters with zeros
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)      # Gradient descent
    w , b= parameters["w"],parameters["b"]              # Retrieve parameters w and b from dictionary "parameters"
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "num_px" : train_set_x_orig.shape[1],
         "num_py" : train_set_x_orig.shape[2],
         "classes" : classes}
    return d


# 对图片的输入进行处理，主要包括向量化和归一化
def dealimg(train_set_x_orig,test_set_x_orig):
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    train_set_x=train_set_x_flatten/255
    test_set_x=test_set_x_flatten/255
    return train_set_x,test_set_x

# GRADED FUNCTION: sigmoid
def sigmoid(z):    
    s = 1/(1+np.exp(-z))
    return s

# 初始化模型参数
def initialize_with_zeros(dim):
    w = np.zeros((dim,1),dtype=np.float64)
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# GRADED FUNCTION: propagate：正向传播
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)                                    # compute activation
    cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m                                # compute cost
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost

# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b} 
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

# 预测
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction
