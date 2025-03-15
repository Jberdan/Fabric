# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

# Principal Component Analysis (PCA)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# In many real-world data scenarios, we deal with high-dimensional data that can be hard to work with. When used for exploration, PCA helps in reducing the number of variables while retaining most of the original information. This makes the data easier to work with and less resource-intensive for machine learning algorithms.

# CELL ********************

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the penguins dataset
penguins = pd.read_csv('https://raw.githubusercontent.com/MicrosoftLearning/dp-data/main/penguins.csv')

# Remove missing values
penguins = penguins.dropna()

# Prepare the data and target
X = penguins.drop('Species', axis=1)
y = penguins['Species']

# Initialize and apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data
plt.figure(figsize=(8, 6))
for color, target in zip(['navy', 'turquoise', 'darkorange'], penguins['Species'].unique()):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], color=color, alpha=.8, lw=2,
                label=target)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Penguins dataset')
plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The results show a separation between penguins of different species. That is, points of the same color (species) are closer together and points of different colors are further apart. Differences in the feature distributions for different classes result in this separation, suggesting that we can distinguish these species based on their attributes.
