from model_package import *

# 训练逻辑回归模型
def logistic_regression(X,Y,isplot):
    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y.T);
    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
        '% ' + "(percentage of correctly labelled datapoints)")
    if isplot==True:
        plt.show()
