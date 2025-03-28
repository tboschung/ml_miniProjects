import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def transform_data(X):
    """
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1
    """
    X_input = np.hstack([X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))])

    assert X_input.shape == (700, 21)
    return X_input


def fit(X, y):
    weights = np.zeros((21,))
    X_input = transform_data(X)

    weights = LinearRegression(fit_intercept=False).fit(X_input, y).coef_

    assert weights.shape == (21,)
    return weights


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    print(data.head())

    X = data.to_numpy()

    
    w = fit(X, y)
    np.savetxt("./results.csv", w, fmt="%.12f")
