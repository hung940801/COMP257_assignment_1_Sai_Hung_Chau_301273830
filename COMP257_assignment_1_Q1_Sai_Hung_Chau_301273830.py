import matplotlib.pyplot as plt
import random

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA

# Load the MNIST dataset
mnist = fetch_openml("mnist_784")

# Display a few random digits
n_digits = 10
fig, axes = plt.subplots(1, n_digits, figsize=(10, 2))

for i in range(n_digits):
    # Generate a random index within the valid range
    random_idx = random.randint(0, len(mnist.data) - 1)

    digit_image = mnist.data.iloc[random_idx].to_numpy().reshape(28, 28)
    axes[i].imshow(digit_image, cmap=plt.cm.binary)
    axes[i].axis('off')

plt.show()

# Create a PCA instance
pca = PCA()

# Fit and transform the data
mnist_pca = pca.fit_transform(mnist.data)

# Output explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.scatter(mnist_pca[:, 0], mnist_pca[:, 1], c=mnist.target.astype(int), cmap=plt.cm.get_cmap('jet', 10), marker='.')
plt.colorbar(label="Digit Label")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Projection of First and Second Principal Components")
plt.show()

# Create an Incremental PCA instance with desired dimensions
n_components = 154
ipca = IncrementalPCA(n_components=n_components)

# Fit and transform the data
mnist_ipca = ipca.fit_transform(mnist.data)

# Choose a random digit index
random_idx = random.randint(0, len(mnist.data) - 1)

# Original digit
original_digit = mnist.data.iloc[random_idx].to_numpy().reshape(28, 28)

# Compressed digit
compressed_digit = ipca.inverse_transform(mnist_ipca[random_idx]).reshape(28, 28)

# Display the original and compressed digits
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(original_digit, cmap=plt.cm.binary)
axes[0].set_title("Original Digit")
axes[0].axis('off')
axes[1].imshow(compressed_digit, cmap=plt.cm.binary)
axes[1].set_title("Compressed Digit")
axes[1].axis('off')
plt.show()
