import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Matrix initialization
A = np.array([[1, 2, 3, 4],
              [5, 5, 6, 7],
              [1, 4, 2, 3],
              [5, 3, 2, 1],
              [8, 1, 2, 2]])

# Create a DataFrame
df = pd.DataFrame(A, columns=['f1', 'f2', 'f3', 'f4'])

# Standardize the DataFrame
df_std = (df - df.mean()) / df.std()

# Calculate covariance matrix
cov_matrix = np.cov(df_std.T, bias=False)

# Eigen decomposition of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Select top n_components
n_components = 3
top_eigen_vectors = eigen_vectors[:, :n_components]

# Project data to the new space using the top eigenvectors
transformed_data = np.dot(df_std, top_eigen_vectors)

# PCA using sklearn for verification
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_std)

# Create DataFrame for principal components
principal_df = pd.DataFrame(data=principal_components, 
                            columns=[f'principal component {i+1}' for i in range(n_components)])

# Output
print("Transformed Data (Manual PCA):")
print(transformed_data)

print("\nTransformed Data (Sklearn PCA):")
print(principal_df)
