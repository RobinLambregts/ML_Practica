import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1).values
y = data['target'].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5
    cost = -1/m * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f'Iteration {i}: Cost = {cost}')

    return weights, cost_history

X_train_intercept = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_intercept = np.c_[np.ones(X_test.shape[0]), X_test]

weights = np.zeros(X_train_intercept.shape[1])

learning_rate = 0.01
iterations = 1000

weights, cost_history = gradient_descent(X_train_intercept, y_train, weights, learning_rate, iterations)

y_pred_prob = sigmoid(np.dot(X_test_intercept, weights))

y_pred = [1 if i >= 0.5 else 0 for i in y_pred_prob]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
