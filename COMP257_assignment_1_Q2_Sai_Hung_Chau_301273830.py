import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import make_swiss_roll
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

# Generate the Swiss roll dataset
X, _ = make_swiss_roll(n_samples=1000, noise=0.2, random_state=30)

y = np.digitize(X[:, 2], bins=[-10, 5, 10, 15])

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("Swiss Roll Dataset")
plt.colorbar()
plt.show()

# Apply Kernel PCA with linear kernel
kpca_linear = KernelPCA(kernel='linear', n_components=2)
X_kpca_linear = kpca_linear.fit_transform(X)

# Apply Kernel PCA with RBF kernel
kpca_rbf = KernelPCA(kernel='rbf', n_components=2)
X_kpca_rbf = kpca_rbf.fit_transform(X)

# Apply Kernel PCA with sigmoid kernel
kpca_sigmoid = KernelPCA(kernel='sigmoid', n_components=2)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("kPCA with Linear Kernel")

plt.subplot(132)
plt.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("kPCA with RBF Kernel")

plt.subplot(133)
plt.scatter(X_kpca_sigmoid[:, 0], X_kpca_sigmoid[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("kPCA with Sigmoid Kernel")

plt.tight_layout()
plt.show()

# Create a pipeline with kPCA and Logistic Regression
pipe = Pipeline([
    ('kpca', KernelPCA(kernel='rbf', gamma=0.04, eigen_solver='auto', n_components=2)),
    ('logreg', LogisticRegression(max_iter=10000))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': np.linspace(0.03, 0.05, 10)
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X, y)

# Print best parameters
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Evaluate the model on the data
y_pred = grid_search.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy on the dataset:", accuracy)

# Get the results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

# Pivot the results table for plotting
pivot_table = results.pivot(index='param_kpca__gamma', columns='param_kpca__kernel', values='mean_test_score')

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f', cbar=True)
plt.title("GridSearchCV Results")
plt.xlabel("Kernel")
plt.ylabel("Gamma")
plt.show()