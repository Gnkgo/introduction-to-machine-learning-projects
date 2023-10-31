'''
The task of the machine learning in this code is to perform Ridge regression with 
cross-validation on a given dataset. Ridge regression is used to model the relationship 
between input features and labels while preventing overfitting. 
The goal is to find the optimal regularization parameter (lambda) for Ridge regression, 
which results in the lowest RMSE (Root Mean Square Error) when making predictions on unseen data. 
The code's objective is to determine which value of lambda provides the best trade-off between bias 
and variance in the model. Ultimately, it seeks to build a predictive model that minimizes prediction errors.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


def fit(X, y, lam):

    w = np.zeros((13,))
    #The Ridge model calculates minwn∑i=1(yi−wTxi)2+λ||w||22. We only use one lambda at a time
    model = Ridge(alpha = lam, fit_intercept = False)
    #We fit the data
    model.fit(X, y)
    #return the coefficients of the Ridge regression --> Also referred to as the weights
    w = model.coef_
    assert w.shape == (13,)
    return w


def calculate_RMSE(w, X, y):
    RMSE = 0
    #Matrix multiplication
    y_pred = X @ w
    #We already got the weights -> we already caluclated Ridge. We only have to multiply wiht X now to get the prediction. 
    RMSE = mean_squared_error(y, y_pred)**0.5
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    kv = KFold(n_splits = n_folds, shuffle = True, random_state = 42)
    #we use shuffle = True to guarantee that the dataset is not sorted in any kind of way.
    #we use the random_state so we can replicate our results
    for i, (train_index, test_index) in enumerate(kv.split(X)):
        
        #train our dataset --> iterate through it 
        X_train, y_train = X[train_index], y[train_index]

        #test our dataset
        X_test, y_test = X[test_index], y[test_index]

        for j, lam in enumerate(lambdas):
            #for each lambda use the fit function we implemted beforehand
            w = fit(X_train, y_train, lam)
            #insert it in our matrix so we can get the average in the end
            RMSE_mat[i][j] = calculate_RMSE(w, X_test, y_test)

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE



if __name__ == "__main__":
    # Data loading
    data = pd.read_csv('train.csv')
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")