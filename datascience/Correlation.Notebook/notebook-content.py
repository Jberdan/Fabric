# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# MARKDOWN ********************

# # Correlation
# Correlation is a statistical method used to evaluate the strength and direction of the linear relationship between two quantitative variables. The correlation coefficient ranges from -1 to 1.

# CELL ********************

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the penguins dataset
penguins = pd.read_csv('https://raw.githubusercontent.com/MicrosoftLearning/dp-data/main/penguins.csv')

# Calculate the correlation matrix
corr = penguins.corr()

# Create a heatmap
sns.heatmap(corr, annot=True)
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The strongest correlation in the dataset is between FlipperLength and BodyMass variables, with a correlation coefficient of 0.87. This suggests that penguins with larger flippers tend to have a larger body mass.
