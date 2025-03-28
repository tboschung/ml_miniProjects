
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def fit(X, y, lam):
    weights = np.zeros((13,))

    model = Ridge(alpha=lam, fit_intercept=False)
    model.fit(X, y)
    weights = model.coef_

    assert weights.shape == (13,)
    return weights


def calculate_RMSE(w, X, y):
    rmse = 0
    y_pred = X @ w
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    assert np.isscalar(rmse)
    return rmse


def average_LR_RMSE(X, y, lambdas, n_folds):
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    kf = KFold(n_splits=n_folds)
    
    #loop over all 10 splits
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit split for all lambdas
        for j, lam in enumerate(lambdas):
            weights = fit(X_train, y_train, lam)
            RMSE_mat[i][j] = calculate_RMSE(weights, X_test, y_test)

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    X = data.to_numpy()
    lambdas = [0.1, 0.5, 20, 300, 1000]
    n_folds = 8
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")