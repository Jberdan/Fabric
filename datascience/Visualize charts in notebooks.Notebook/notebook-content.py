# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# MARKDOWN ********************

# # Visualize charts in notebooks


# MARKDOWN ********************

# Summary statistics

# CELL ********************

import pandas as pd

df = pd.DataFrame({
    'Height_in_cm': [170, 180, 175, 185, 178],
    'Weight_in_kg': [65, 75, 70, 80, 72],
    'Age_in_years': [25, 30, 28, 35, 32]
})

desc_stats = df.describe()
print(desc_stats)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Univariate analysis

# MARKDOWN ********************

# Univariate analysis is the simplest form of data analysis where the data being analyzed contains only one variable. The main purpose of univariate analysis is to describe the data and find patterns that exist within it.

# CELL ********************

import numpy as np
import matplotlib.pyplot as plt

# Let's assume these are the heights of a group in inches
heights_in_inches = [63, 64, 66, 67, 68, 69, 71, 72, 73, 55, 75]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Boxplot
axs[0].boxplot(heights_in_inches, whis=0.5)
axs[0].set_title('Box plot of heights')

# Histogram
bins = range(min(heights_in_inches), max(heights_in_inches) + 5, 5)
axs[1].hist(heights_in_inches, bins=bins, alpha=0.5)
axs[1].set_title('Frequency distribution of heights')
axs[1].set_xlabel('Height (in)')
axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# These are a few conclusions we can draw from the results.
# 
# In the box plot, the distribution of heights is skewed to the left, meaning there are many individuals with heights significantly below the mean.
# There are two potential outliers: 55 inches (4’7") and 75 inches (6’3"). These values are lower and higher than the rest of the data points.
# The distribution of heights is roughly symmetrical around the median, assuming that the outliers don't significantly skew the distribution.

# MARKDOWN ********************

# Bivariate and multivariate analysis

# MARKDOWN ********************

# Scatter plots

# CELL ********************

import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(0)  # for reproducibility
house_sizes = np.random.randint(1000, 3000, size=50)  # Size of houses in square feet
house_prices = house_sizes * 100 + np.random.normal(0, 20000, size=50)  # Price of houses in dollars

# Create scatter plot
plt.scatter(house_sizes, house_prices)

# Set plot title and labels
plt.title('House Prices vs Size')
plt.xlabel('Size (sqt)')
plt.ylabel('Price ($)')

# Display the plot
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# In this scatter plot, each point represents a house. You see that as the size of the house increases (moving right along the x-axis), the price also tends to increase (moving up along the y-axis).
# 
# This type of analysis helps us understand how changes in the dependent variables affect the target variable. By analyzing the relationships between these variables, we can make predictions about the target variable based on the values of the dependent variables.
# 
# Moreover, this analysis can help identify which dependent variables have a significant impact on the target variable. This is useful for feature selection in machine learning models, where the goal is to use the most relevant features to predict the target.

# MARKDOWN ********************

# Line plot

# CELL ********************

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import matplotlib.dates as mdates

# Generate monthly dates from 2020 to 2022
dates = [datetime(2020, 1, 1) + timedelta(days=30*i) for i in range(36)]

# Generate corresponding house prices with some randomness
prices = [200000 + 5000*i + random.randint(-5000, 5000) for i in range(36)]

plt.figure(figsize=(10,5))

# Plot data
plt.plot(dates, prices)

# Format x-axis to display dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6)) # set interval to 6 months
plt.gcf().autofmt_xdate() # Rotation

# Set plot title and labels
plt.title('House Prices Over Years')
plt.xlabel('Year-Month')
plt.ylabel('House Price ($)')

# Show the plot
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Line plots are simple to understand and read. They provide a clear, high-level overview of the data’s progression over time, making them a popular choice for time series data.


# MARKDOWN ********************

# Pair plot


# MARKDOWN ********************

# A pair plot can be useful when you want to visualize the relationship between multiple variables at once.

# CELL ********************

import seaborn as sns
import pandas as pd

# Sample data
data = {
    'Size': [1000, 2000, 3000, 1500, 1200],
    'Bedrooms': [2, 4, 3, 2, 1],
    'Age': [5, 10, 2, 6, 4],
    'Price': [200000, 400000, 350000, 220000, 150000]
}

df = pd.DataFrame(data)

# Create a pair plot
sns.pairplot(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# This creates a grid of plots where each feature is plotted against every other feature. On the diagonal are histograms showing the distribution of each feature. The off-diagonal plots are scatter plots showing the relationship between two features.
# 
# This kind of visualization can help us understand how different features are related to each other and could potentially be used to inform decisions about buying or selling houses.
