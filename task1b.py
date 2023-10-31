'''
The primary task of this code is to build and evaluate regression models 
using Elastic Net regularization and feature transformation to predict the 
target variable "y" based on the input features. 
The code then exports the optimal model's coefficients as the final results.
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold



def transform_data(X):

    #Get the transformed matrix with all the different functions
    X_transformed = np.zeros((700, 21))
    X_transformed[:, :5] = X
    X_transformed[:, 5:10] = X**2
    X_transformed[:, 10:15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20] = 1
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    w = np.zeros((21,))
    X_transformed = transform_data(X)

    # Folds
    n_folds = 10    
    lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400]
    best_err = 1000000000
    best_lambda = None
    model = None
    lam = None
    ratio = 0.3


    kv = KFold(n_splits = n_folds, shuffle = True, random_state = 42)
    #we use shuffle = True to guarantee that the dataset is not sorted in any kind of way.
    #we use the random_state so we can replicate our results

    for i, lam in enumerate(lambdas):
        RMSE_error_temp = 0
        model = ElasticNet(alpha=lam, l1_ratio=ratio, fit_intercept=False)

        for i, (train_index, test_index) in enumerate(kv.split(X_transformed)):
            #train our dataset --> iterate through it 
            X_train, y_train = X_transformed[train_index], y[train_index]

            #test our dataset
            X_test, y_test = X_transformed[test_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            err = np.sqrt(mean_squared_error(y_test, y_pred))**0.5
            RMSE_error_temp += err
        
        RMSE_error_temp /= len(lambdas)
        print("RMSE ", RMSE_error_temp, " Lam ", lam)
        if (RMSE_error_temp < best_err):
            best_err = RMSE_error_temp
            best_lambda = lam
            w = model.coef_

    #only to calculate mean squared error
    #X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    #sgd_reg.fit(X_train, y_train)
    #y_pred = sgd_reg.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)**0.5
    #print("Mean squared error:", mse)

    #get the weight with the coefficients
    print(best_lambda, best_err)
    assert w.shape == (21,)
    return w

# Main function
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
