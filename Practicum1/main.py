import numpy as np
import csv
from sklearn.linear_model import Ridge

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    medv = data[:, -1]
    rest = data[:, :-1]
    return medv, rest


def testLoadingData(file):
    print(F"{load_data(file)}")

def train_ridge_regression(X, y, alpha=1.0):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term (intercept)
    ridge_reg = Ridge(alpha=alpha, solver="cholesky")  # Use Cholesky solver
    ridge_reg.fit(X_b, y)
    return ridge_reg.coef_

def train_linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term (intercept)
    theta = train_ridge_regression(restDataTraining, medvDataTraining)
    return theta


def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term (intercept)
    return X_b.dot(theta)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    medvDataTest, restDataTest = load_data('./BostonHousing.csv')
    medvDataTraining, restDataTraining = load_data('./BostonHousing2.csv')

    theta = train_linear_regression(restDataTraining, medvDataTraining)

    # Predictions and MSE for training data
    y_train_pred = predict(restDataTraining, theta)
    train_mse = mean_squared_error(medvDataTraining, y_train_pred)
    print(f"Training Mean Squared Error: {train_mse}")

    # Predictions and MSE for test data
    y_test_pred = predict(restDataTest, theta)
    test_mse = mean_squared_error(medvDataTest, y_test_pred)
    print(f"Testing Mean Squared Error: {test_mse}")
