'''
In summary, the primary task of this code is to develop a predictive model 
for estimating prices based on input features, using Gaussian Process Regression with kernel selection 
and cross-validation to determine the best-performing model for prediction.
'''

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF, RationalQuadratic, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold




# Function for data loading and preprocessing
def data_loading():
    # Load training data
    train_df = pd.read_csv("train.csv")
    # Load test data
    test_df = pd.read_csv("test.csv")

    # Drop all rows where 'price_CHF' is null since we need to predict that value.
    train_df = train_df[train_df['price_CHF'].notna()]

    # Initialize the X_train, X_test, and y_train
    X_train = train_df.drop(['price_CHF'], axis=1)
    y_train = train_df['price_CHF']
    X_test = test_df

    # Create binary columns for 'season' feature
    X_train['is_spring'] = X_train.apply(lambda row: 1.0 if row['season'] == 'spring' else 0.0, axis=1)
    X_train['is_summer'] = X_train.apply(lambda row: 1.0 if row['season'] == 'summer' else 0.0, axis=1)
    X_train['is_winter'] = X_train.apply(lambda row: 1.0 if row['season'] == 'winter' else 0.0, axis=1)
    X_train['is_autumn'] = X_train.apply(lambda row: 1.0 if row['season'] == 'autumn' else 0.0, axis=1)
    X_train = X_train.drop(['season'], axis=1)

    X_test['is_spring'] = X_test.apply(lambda row: 1.0 if row['season'] == 'spring' else 0.0, axis=1)
    X_test['is_summer'] = X_test.apply(lambda row: 1.0 if row['season'] == 'summer' else 0.0, axis=1)
    X_test['is_winter'] = X_test.apply(lambda row: 1.0 if row['season'] == 'winter' else 0.0, axis=1)
    X_test['is_autumn'] = X_test.apply(lambda row: 1.0 if row['season'] == 'autumn' else 0.0, axis=1)
    X_test = X_test.drop(['season'], axis=1)

    # Use SimpleImputer to fill NaN values with the mean of the feature
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train.to_numpy(), X_test

# Function to calculate R-squared (R2) score
def r2(y_pred, y):
    R2 = r2_score(y, y_pred)
    assert np.isscalar(R2)
    return R2

# Function to perform kernel-based model fitting and prediction
def kfit(kv, X_test, X_train, y_train):
    models = []
    R2_arr = np.zeros((kv.get_n_splits()))

    for i, (train_index, test_index) in enumerate(kv.split(X_train)):
        # Train our dataset
        X_train_folds, y_train_folds = X_train[train_index], y_train[train_index]
        # Test our dataset
        X_test_folds, y_test_folds = X_train[test_index], y_train[test_index]
        # Fit with a given kernel
        gpr = GaussianProcessRegressor(kernel=RBF() + RationalQuadratic() + RBF() + WhiteKernel(), random_state=42)
        gpr.fit(X_train_folds, y_train_folds)
        y_pred = gpr.predict(X_test_folds)
        models.append(gpr)
        R2_arr[i] = r2(y_pred, y_test_folds)

    ind = np.argmax(R2_arr)
    fit = models[ind]
    y_pred = fit.predict(X_test)

    return y_pred

# Function for modeling and prediction
def modeling_and_prediction(X_train, y_train, X_test):
    n_folds = 10
    y_test = kfit(KFold(n_splits=n_folds, shuffle=True, random_state=42), X_test, X_train, y_train)
    return y_test

# Main function
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # Modeling and prediction
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
