import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders import BinaryEncoder
from sklearn.metrics import root_mean_squared_error

def load_salary(salary_file):
    df = pd.read_csv(salary_file)

    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    binary_encoder = BinaryEncoder(cols=categorical_columns)
    df = binary_encoder.fit_transform(df)

    X = df.drop('Salary', axis=1)  # features

    # Split the data into training and testing sets
    y = df['Salary']  # target variable

    return X, y


def load_life_ex(life_ex_file):
    df = pd.read_csv(life_ex_file)

    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    binary_encoder = BinaryEncoder(cols=categorical_columns)
    df = binary_encoder.fit_transform(df)

    X = df.drop('Life expectancy ', axis=1)  # features

    # Split the data into training and testing sets
    y = df['Life expectancy ']  # target variable

    return X, y


def run_regression(X, y, degree=2):
    # Split the data into training and testing sets (5% test data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Impute missing values with the mean
    simple_imputer = SimpleImputer(strategy='mean')
    X_train = simple_imputer.fit_transform(X_train)
    X_test = simple_imputer.transform(X_test)

    # Reshape target variables for imputation
    y_train = simple_imputer.fit_transform(y_train.values.reshape(-1, 1))
    y_test = simple_imputer.transform(y_test.values.reshape(-1, 1))

    # Step 1: Apply PolynomialFeatures to create polynomial terms (degree 2 by default)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)  # Transform the training data
    X_test_poly = poly.transform(X_test)  # Transform the test data

    # Step 2: Train the Linear Regression model with polynomial features
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)  # Fit the model to the transformed training data

    # Step 3: Predict and calculate Root Mean Squared Error (RMSE) on the training data
    y_train_pred = lr.predict(X_train_poly)  # Predict on the training data
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    print(f"Training Root Mean Squared Error: {train_rmse}")

    # Step 4: Predict and calculate RMSE on the testing data
    y_test_pred = lr.predict(X_test_poly)  # Predict on the testing data
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    print(f"Testing Root Mean Squared Error: {test_rmse}")
    return 0


def main(salary_file, life_ex_file):
    # Load the data
    X_salary, y_salary = load_salary(salary_file)
    X_life, y_life = load_life_ex(life_ex_file)

    print("Life expectancy data:")
    run_regression(X_life, y_life)

    print("Salary data:")
    run_regression(X_salary, y_salary)


if __name__ == "__main__":
    main('./Salary_Data.csv', './Life Expectancy Data.csv')