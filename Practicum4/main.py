import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class MySVM:
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None

    def fit(self, X, y):
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)


# Load dataset
data = pd.read_csv('moonDataset.csv')
X = data.drop('label', axis=1).values
y = data['label'].values

# Convert labels to -1 and 1
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create instances of MySVM for Linear and RBF kernels
clf_linear = MySVM(kernel='linear')
clf_linear.fit(X_train, y_train)
predictions_linear = clf_linear.predict(X_test)

clf_rbf = MySVM(kernel='rbf', gamma=0.5)
clf_rbf.fit(X_train, y_train)
predictions_rbf = clf_rbf.predict(X_test)


# Calculate accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


print("Linear SVM classification accuracy:", accuracy(y_test, predictions_linear))
print("RBF SVM classification accuracy:", accuracy(y_test, predictions_rbf))


# 3D Visualization of the dataset with decision boundaries
def visualize_svm_3d(clf, X, y, title="SVM Decision Boundary in 3D"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y, cmap=plt.cm.coolwarm)

    # Create a grid to plot the decision boundary
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid for the first two dimensions
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 100),
                         np.linspace(x1_min, x1_max, 100))

    # Calculate Z values for the decision boundary and margins
    Z = np.zeros(xx.shape)
    Z_margin_pos = np.zeros(xx.shape)
    Z_margin_neg = np.zeros(xx.shape)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            # Predict the Z coordinate based on the first two dimensions
            sample = np.array([[xx[i, j], yy[i, j], 0]])  # Use a fixed value for the third dimension
            Z[i, j] = clf.decision_function(sample)[0]
            Z_margin_pos[i, j] = Z[i, j] + 1  # Margin for positive class
            Z_margin_neg[i, j] = Z[i, j] - 1  # Margin for negative class

    # Plot the decision boundary
    ax.contour3D(xx, yy, Z, levels=[0], colors='k', linestyles='solid', linewidths=2)
    # Plot the margins
    ax.contour3D(xx, yy, Z_margin_pos, levels=[0], colors='r', linestyles='dashed', linewidths=1)
    ax.contour3D(xx, yy, Z_margin_neg, levels=[0], colors='b', linestyles='dashed', linewidths=1)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Decision Function Value')
    ax.set_title(title)
    ax.set_zlim(-1, 1)

    plt.show()


# Visualize both models in 3D
visualize_svm_3d(clf_linear, X, y, title="Linear-based SVM")
visualize_svm_3d(clf_rbf, X, y, title="Kernel-based SVM")
