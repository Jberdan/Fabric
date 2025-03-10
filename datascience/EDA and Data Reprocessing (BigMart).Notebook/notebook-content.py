# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "296e29e0-1c40-4d91-b5b6-89e6736c6191",
# META       "default_lakehouse_name": "datasciencelearning",
# META       "default_lakehouse_workspace_id": "a601866f-f379-44b7-b085-0e5d028917d2"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # **BigMart Sales Prediction**

# MARKDOWN ********************

# -----------------------------
# ## **Context**
# -----------------------------
# 
# In today’s modern world, huge shopping centers such as big malls and marts are recording data related to sales of items or products as an important step to predict the sales and get an idea about future demands that can help with inventory management. Understanding what role certain properties of an item play and how they affect their sales is imperative to any retail business
# 
# The Data Scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. Using this data, **BigMart is trying to understand the properties of products and stores which play a key role in increasing sales**.
# 
# -----------------------------
# ## **Objective** 
# -----------------------------
# 
# To build a predictive model that can find out the sales of each product at a particular store and then provide actionable recommendations to the BigMart sales team to understand the properties of products and stores which play a key role in increasing sales.
# 
# -----------------------------
# ## **Dataset** 
# -----------------------------
# 
# - Item_Identifier : Unique product ID
# 
# - Item_Weight : Weight of the product
# 
# - Item_Fat_Content : Whether the product is low fat or not
# 
# - Item_Visibility : The % of the total display area of all products in a store allocated to the particular product
# 
# - Item_Type : The category to which the product belongs
# 
# - Item_MRP : Maximum Retail Price (list price) of the product
# 
# - Outlet_Identifier : Unique store ID
# 
# - Outlet_Establishment_Year : The year in which the store was established
# 
# - Outlet_Size : The size of the store in terms of ground area covered
# 
# - Outlet_Location_Type : The type of city in which the store is located
# 
# - Outlet_Type : Whether the outlet is just a grocery store or some sort of supermarket
# 
# - Item_Outlet_Sales : Sales of the product in the particular store. This is the outcome variable to be predicted.
# 
# We have two datasets - train (8523) and test (5681) data. The training dataset has both input and output variable(s). You will need to predict the sales for the test dataset.
# 
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.


# MARKDOWN ********************

# ## **Importing the libraries and overview of the dataset**

# CELL ********************

# Importing libraries for data manipulation
import numpy as np

import pandas as pd

# Importing libraries for data visualization
import seaborn as sns

import matplotlib.pyplot as plt

# Importing libraries for building linear regression model
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importing libraries for scaling the data
from sklearn.preprocessing import MinMaxScaler

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Note: The first section of the notebook is the section that has been covered in the previous case studies. For this discussion, this part can be skipped and we can directly refer to this **<a href = #link1>summary</a>** of data description and observations from EDA.

# MARKDOWN ********************

# ### **Loading the datasets**

# CELL ********************

# Load data into pandas DataFrame from "/lakehouse/default/Files/EDA and Data Preprocessing/bigMartSales.csv"
train_df = pd.read_csv("/lakehouse/default/Files/EDA and Data Preprocessing/bigMartSales.csv")
display(train_df)

# Load data into pandas DataFrame from "/lakehouse/default/Files/EDA and Data Preprocessing/BM_Test.csv"
test_df = pd.read_csv("/lakehouse/default/Files/EDA and Data Preprocessing/BM_Test.csv")
display(test_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Checking the first 5 rows of the dataset
train_df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - As the variables **Item_Identifier** and **Outlet_Identifier** are only ID variables, we assume that they don't have any predictive power to predict the dependent variable - **Item_Outlet_Sales**.
# - We will remove these two variables from both the datasets.

# CELL ********************

train_df = train_df.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1)

test_df = test_df.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, let's find out some more information about the training dataset, i.e., the total number of observations in the dataset, columns and their data types, etc.

# MARKDOWN ********************

# ### **Checking the info of the training data**

# CELL ********************

train_df.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - The train dataset has **8523 observations and 10 columns**. 
# - Two columns **Item_Weight** and **Outlet_Size** have missing values as the number of non-null values is less the total number of observations for these two variables.
# - We observe that the columns **Item_Fat_Content**, **Item_Type**, **Outlet_Size**, **Outlet_Location_Type**, and **Outlet_Type** have data type **object**, which means they are strings or categorical variables.
# - The remaining variables are all numerical in nature.

# MARKDOWN ********************

# As we have already seen, two columns have missing values in the dataset. Let's check the **percentage of missing values** using the below code.

# CELL ********************

(train_df.isnull().sum() / train_df.shape[0])*100

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The percentage of missing values for the columns **Item_Weight** and **Outlet_Size** is ~17% and ~28% respectively. Now, let's see next how to treat these missing values.

# MARKDOWN ********************

# ## **EDA and Data Preprocessing**

# MARKDOWN ********************

# Now that we have an understanding of the business problem we want to solve, and we have loaded the datasets, the next step to follow is to have a better understanding of the dataset, i.e., what is the distribution of the variables, what are different relationships that exist between variables, etc. If there are any data anomalies like missing values or outliers, how do we treat them to prepare the dataset for building the predictive model?

# MARKDOWN ********************

# ### **Univariate Analysis**

# MARKDOWN ********************

# Let us now start exploring the dataset, by performing univariate analysis of the dataset, i.e., analyzing/visualizing the dataset by taking one variable at a time. Data visualization is a very important skill to have and we need to decide what charts to plot to better understand the data. For example, what chart makes sense to analyze categorical variables or what charts are suited for numerical variables, and also it depends on what relationship between variables we want to show.

# MARKDOWN ********************

# Let's start with analyzing the **categorical** variables present in the data. There are five categorical variables in this dataset and we will create univariate bar charts for each of them to check their distribution. 

# MARKDOWN ********************

# Below we are creating 4 subplots to plot four out of the five categorical variables in a single frame.

# CELL ********************

fig, axes = plt.subplots(2, 2, figsize = (18, 10))
  
fig.suptitle('Bar plot for all categorical variables in the dataset')

sns.countplot(ax = axes[0, 0], x = 'Item_Fat_Content', data = train_df, color = 'blue', 
              order = train_df['Item_Fat_Content'].value_counts().index);

sns.countplot(ax = axes[0, 1], x = 'Outlet_Size', data = train_df, color = 'blue', 
              order = train_df['Outlet_Size'].value_counts().index);

sns.countplot(ax = axes[1, 0], x = 'Outlet_Location_Type', data = train_df, color = 'blue', 
              order = train_df['Outlet_Location_Type'].value_counts().index);

sns.countplot(ax = axes[1, 1], x = 'Outlet_Type', data = train_df, color = 'blue', 
              order = train_df['Outlet_Type'].value_counts().index);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - From the above univariate plot for the variable **Item_Fat_Content**, it seems like there are errors in the data. The category - **Low Fat** is also written as **low fat** and **LF**. Similarly, the category **Regular** is also written as **reg** sometimes. So, **we need to fix this issue in the data**.
# - For the column **Outlet_Size**, we can see that the most common category is - **Medium** followed by Small and High.
# - The most common category for the column **Outlet_Location_Type** is **Tier 3**, followed by Tier 2 and Tier 1. This makes sense now if we combine this information with the information on column **Outlet_Size**. We would expect High outlet size stores to be present in Tier 1 cities and the count of tier 1 cities is less so the count of high outlets size is also less. And we would expect more medium and small outlet sizes in the dataset because we have more number outlets present in tier 3 and tier 2 cities in the dataset.
# - In the column **Outlet_Type**, the majority of the stores are of **Supermarket Type 1** and we have a less and almost equal number of representatives in the other categories - Supermarket Type 2, Supermarket Type 3, and Grocery Store.


# MARKDOWN ********************

# Below we are analyzing the categorical variable **Item_Type**.

# CELL ********************

fig = plt.figure(figsize = (18, 6))

sns.countplot(x = 'Item_Type', data = train_df, color = 'blue', order = train_df['Item_Type'].value_counts().index);

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - From the above plot, we observe that majority of the items sold in these stores are **Fruits and Vegetables**, followed by **Snack Foods** and **Household items**.

# MARKDOWN ********************

# Before we move ahead with the univariate analysis for the numerical variables, let's first fix the data issues that we have found for the column **Item_Fat_Content**.

# MARKDOWN ********************

# In the below code, we are replacing the categories - **low fat** and **LF** by **Low Fat** using lambda function and also we are replacing the category **reg** by Regular.

# CELL ********************

train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x == 'low fat' or x == 'LF' else x)

train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Regular' if x == 'reg' else x)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The data preparation steps that we do on the training data, we need to perform the same steps on the test data as well. So, below we are performing the same transformations on the **test** dataset. 

# CELL ********************

test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x == 'low fat' or x == 'LF' else x)

test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Regular' if x == 'reg' else x)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Below we are analyzing all the **numerical** variables present in the data. And since we want to visualize one numerical variable at a time, histogram is the best choice to visualize the data.

# CELL ********************

fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  
fig.suptitle('Histogram for all numerical variables in the dataset')
  
sns.histplot(x = 'Item_Weight', data = train_df, kde = True, ax = axes[0]);

sns.histplot(x = 'Item_Visibility', data = train_df, kde = True, ax = axes[1]);

sns.histplot(x = 'Item_MRP', data = train_df, kde = True, ax = axes[2]);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### **Observations:**

# MARKDOWN ********************

# - The variable **Item_Weight** is approx uniformly distributed and when we will impute the missing values for this column, we will need to keep in mind that we don't end up changing the distribution significantly after imputing those missing values.
# - The variable **Item_Visibility** is a right-skewed distribution which means that there are certain items whose percentage of the display area is much higher than the other items.
# - The variable **Item_MRP** is following an approx multi-modal normal distribution.

# MARKDOWN ********************

# ### **Bivariate Analysis**

# MARKDOWN ********************

# Now, let's move ahead with bivariate analysis to understand how variables are related to each other and if there is a strong relationship between dependent and independent variables present in the training dataset.

# MARKDOWN ********************

# In the below plot, we are analyzing the variables **Outlet_Establishment_Year** and **Item_Outlet_Sales**. Here, since the variable **Outlet_Establishment_Year** is defining a time component, the best chart to analyze this relationship will be a line plot.

# CELL ********************

fig = plt.figure(figsize = (18, 6))

sns.lineplot(x = 'Outlet_Establishment_Year', y = 'Item_Outlet_Sales', data = train_df, ci = None, estimator = 'mean');

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - The average sales are almost constant every year, we don't see any increasing/decreasing trend in sales with time. So, the variable year might not be a good predictor to predict sales, which we can check later in the modeling phase.
# - Also, in the year 1998, the average sales plummeted. This might be due to some external factors which are not included in the data.

# MARKDOWN ********************

# Next, we are trying to find out linear correlations between the variables. This will help us to know which numerical variables are correlated with the target variable. Also, we can find out **multi-collinearity**, i.e., which pair of independent variables are correlated with each other.

# CELL ********************

fig = plt.figure(figsize = (18, 6))

sns.heatmap(train_df.corr(), annot = True);

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - From the above plot, it seems that only the independent variable **Item_MRP** has a moderate linear relationship with the dependent variable **Item_Outlet_Sales**.
# - For the remaining, it does not seem like there is any strong positive/negative correlation between the variables.

# MARKDOWN ********************

# Next, we are creating the bivariate scatter plots to check relationships between the pair of independent and dependent variables.

# CELL ********************

fig, axes = plt.subplots(1, 3, figsize = (20, 6))
  
fig.suptitle('Bi-variate scatterplot for all numerical variables with the dependent variable')
  
sns.scatterplot(x = 'Item_Weight', y = 'Item_Outlet_Sales', data = train_df, ax = axes[0]);

sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = train_df, ax = axes[1]);

sns.scatterplot(x = 'Item_MRP', y = 'Item_Outlet_Sales', data = train_df, ax = axes[2]);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - The first scatter plot shows the data is completely random and there is no relationship between **Item_Weight** and **Item_outlet_Sales**. This is also evident from the correlation value which we got above, i.e., there is no strong correlation between these two variables.
# - The second scatter plot between **Item_Visibility** and **Item_outlet_Sales** shows that there is no strong relationship between them. But we can see a pattern, as the **Item_Visibility** increases from **0.19**, the sales decrease. This might be due to the reason that the management has given more visibility to those items which are not generally sold often, thinking that better visibility would increase the sales. This information can also help us to engineer new features like - a categorical variable with categories - **high visibility** and **low visibility**. But we are not doing that here. This is an idea, we may want to pursue in the future.
# - From the third scatter plot between the variables - **Item_MRP** and **Item_outlet_Sales**, it is clear that there is a positive correlation between them, and the variable **Item_MRP** would have a good predictive power to predict the sales.


# MARKDOWN ********************

# ### <a id='link1'>Summary of EDA</a>
# 
# **Data Description:**
# 
# * We have two datasets - train (8523) and test (5681) data. The training dataset has both input and output variable(s). You will need to predict the sales for the test dataset.
# * The data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
# * As the variables Item_Identifier and Outlet_Identifier are ID variables, we assume that they don't have any predictive power to predict the dependent variable - Item_Outlet_Sales. We will remove these two variables from both datasets.
# * The training dataset has 8523 observations and 10 columns.
# * Two columns Item_Weight and Outlet_Size have missing values as the number of non-null values is less than the total number of observations for these two variables.
# * We observe that the columns Item_Fat_Content, Item_Type, Outlet_Size, Outlet_Location_Type, and Outlet_Type have data type objects, which means they are strings or categorical variables.
# * The remaining variables are all numerical in nature.
# 
# 
# **Observations from EDA:**
# 
# * From the Univariate plot for the variable Item_Fat_Content, it seems like there are errors in the data. The category - Low Fat is also written as low fat and LF. Similarly, the category Regular is also written as reg sometimes. So, we need to fix this issue in the data.
# * For the column Outlet_Size, we can see that the most common category is - Medium followed by Small and High.
# * The most common category for the column Outlet_Location_Type is Tier 3, followed by Tier 2 and Tier 1. This makes sense now if we combine this information with the information on column Outlet_Size. We would expect High outlet size stores to be present in Tier 1 cities and the count of tier 1 cities is less so the count of high outlets size is also less. And we would expect more medium and small outlet sizes in the dataset because we have more number outlets present in tier 3 and tier 2 cities in the dataset.
# * In the column Outlet_Type, the majority of the stores are of Supermarket Type 1 and we have a less and almost equal number of representatives in the other categories - Supermarket Type 2, Supermarket Type 3, and Grocery Store.
# * We observed that majority of the items sold in these stores are Fruits and Vegetables, followed by Snack Foods and Household items.
# * The variable Item_Weight is approx uniformly distributed and when we will impute the missing values for this column, we will need to keep in mind that we don't end up changing the distribution significantly after imputing those missing values.
# * The variable Item_Visibility is a right-skewed distribution which means that there are certain items whose percentage of the display area is much higher than the other items.
# * The variable Item_MRP is following an approx multi-modal normal distribution.
# * The average sales are almost constant every year, we don't see any increasing/decreasing trend in sales with time. So, the variable year might not be a good predictor to predict sales, which we can check later in the modeling phase.
# * Also, in the year 1998, the average sales plummeted. This might be due to some external factors which are not included in the data.
# * It seems that only the independent variable Item_MRP has a moderate linear relationship with the dependent variable Item_Outlet_Sales.
# * For the remaining, it does not seem like there is any strong positive/negative correlation between the variables.
# * The scatter plot between Item_Weight and Item_outlet_Sales shows the data is completely random and there is no relationship between Item_Weight and Item_outlet_Sales. This is also evident from the correlation value which we got above, i.e., there is no strong correlation between these two variables.
# * The scatter plot between Item_Visibility and Item_outlet_Sales shows that there is no strong relationship between them. But we can see a pattern, as the Item_Visibility increases from 0.19, the sales decrease. This might be due to the reason that the management has given more visibility to those items which are not generally sold often, thinking that better visibility would increase the sales. This information can also help us to engineer new features like - a categorical variable with categories - high visibility and low visibility. But we are not doing that here. This is an idea, we may want to pursue in the future.
# * Scatter plot between the variables - Item_MRP and Item_outlet_Sales, shows that there is a positive correlation between them, and the variable Item_MRP would have a good predictive power to predict the sales.


# MARKDOWN ********************

# ### **Missing Value Treatment**

# MARKDOWN ********************

# Here, we are imputing missing values for the variable **Item_Weight**. There are many ways to impute missing values, we can impute the missing values by its **mean**, **median**, and using advanced imputation algorithms like **knn**, etc. But, here we are trying to find out some relationship of the variable **Item_Weight** with other variables in the dataset to impute those missing values.

# MARKDOWN ********************

# And also, after imputing the missing values for the variables, the overall distribution of the variable should not change significantly.

# CELL ********************

fig = plt.figure(figsize = (18, 3))

sns.heatmap(train_df.pivot_table(index = 'Item_Fat_Content', columns = 'Item_Type', values = 'Item_Weight'), annot = True);

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - In the above heatmap, we can see that, based on different combinations of **Item_Types** and **Item_Fat_Content**, the average range of values for the column **Item_Weight** lies between the minimum value of 10 and the maximum value of 14.

# CELL ********************

fig = plt.figure(figsize = (18, 3))

sns.heatmap(train_df.pivot_table(index = 'Outlet_Type', columns = 'Outlet_Location_Type', values = 'Item_Weight'), annot = True);

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - In the above heatmap, we can see that, based on different combinations of **Outlet_Type** and **Outlet_Location_Type**, the average range of values for the column **Item_Weight** is constant at 13.

# CELL ********************

fig = plt.figure(figsize = (18, 3))

sns.heatmap(train_df.pivot_table(index = 'Item_Fat_Content', columns = 'Outlet_Size', values = 'Item_Weight'), annot = True);

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - In the above heatmap, we can see that, based on different combinations of **Item_Fat_Content** and **Outlet_Size**, the average range of values for the column **Item_Weight** is also constant at 13.

# MARKDOWN ********************

# We will impute the missing values using an uniform distribution with parameters a=10 and b=14, as shown below: 

# CELL ********************

item_weight_indices_to_be_updated = train_df[train_df['Item_Weight'].isnull()].index

train_df.loc[item_weight_indices_to_be_updated, 'Item_Weight'] = np.random.uniform(10, 14, 
                                                                                   len(item_weight_indices_to_be_updated))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Performing the same transformation on the **test** dataset.

# CELL ********************

item_weight_indices_to_be_updated = test_df[test_df['Item_Weight'].isnull()].index

test_df.loc[item_weight_indices_to_be_updated, 'Item_Weight'] = np.random.uniform(10, 14, 
                                                                                   len(item_weight_indices_to_be_updated))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Next, we will be imputing missing values for the column **Outlet_Size**. Below, we are creating two different datasets - one where we have non-null values for the column **Outlet_Size** and in the other dataset, all the values of the column **Outlet_Size** are missing. We then check the distribution of the other variables in the dataset where **Outlet_Size** is missing to identify if there is any pattern present or not.

# CELL ********************

outlet_size_data = train_df[train_df['Outlet_Size'].notnull()]

outlet_size_missing_data = train_df[train_df['Outlet_Size'].isnull()]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

fig, axes = plt.subplots(1, 3, figsize = (18, 6))
  
fig.suptitle('Bar plot for all categorical variables in the dataset where the variable Outlet_Size is missing')
  
sns.countplot(ax = axes[0], x = 'Outlet_Type', data = outlet_size_missing_data, color = 'blue', 
              order = outlet_size_missing_data['Outlet_Type'].value_counts().index);

sns.countplot(ax = axes[1], x = 'Outlet_Location_Type', data = outlet_size_missing_data, color = 'blue', 
              order = outlet_size_missing_data['Outlet_Location_Type'].value_counts().index);

sns.countplot(ax = axes[2], x = 'Item_Fat_Content', data = outlet_size_missing_data, color = 'blue', 
              order = outlet_size_missing_data['Item_Fat_Content'].value_counts().index);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - We can see that, wherever **Outlet_Size** is missing in the dataset, the majority of them have **Outlet_Type** as Supermarket Type 1, **Outlet_Location_Type** as Tier 2, and **Item_Fat_Content** as **Low Fat**.

# MARKDOWN ********************

# Now, we are creating a cross-tab of all the above categorical variables against the column **Outlet_Size**, where we want to impute the missing values.

# CELL ********************

fig= plt.figure(figsize = (18, 3))

sns.heatmap(pd.crosstab(index = outlet_size_data['Outlet_Type'], columns = outlet_size_data['Outlet_Size']), annot = True, fmt = 'g')

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:** 
# 
# - We observe from the above heatmap that all the grocery stores have **Outlet_Size** as small.
# - All the Supermarket Type 2 and Supermarket Type 3 have **Outlet_Size** as Medium.

# CELL ********************

fig = plt.figure(figsize = (18, 3))

sns.heatmap(pd.crosstab(index = outlet_size_data['Outlet_Location_Type'], columns = outlet_size_data['Outlet_Size']), annot = True, fmt = 'g')

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - We observe from the above heatmap that all the Tier 2 stores have **Outlet_Size** as Small.

# CELL ********************

fig = plt.figure(figsize = (18, 3))

sns.heatmap(pd.crosstab(index = outlet_size_data['Item_Fat_Content'], columns = outlet_size_data['Outlet_Size']), annot = True, fmt = 'g')

plt.xticks(rotation = 45);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observation:**
# 
# - There does not seem to be any clear pattern between the variables **Item_Fat_Content** and **Outlet_Size**.

# MARKDOWN ********************

# Now, we will use the patterns we have from the variables **Outlet_Type** and **Outlet_Location_Type** to impute the missing values for the column **Outlet_Size**.

# MARKDOWN ********************

# Below, we are identifying the indices in the DataFrame where **Outlet_Size** is null/missing and **Outlet_Type** is Grocery Store, so that we can replace those missing values with the value Small based on the pattern we have identified in above visualizations. Similarly, we are also identifying the indices in the DataFrame where **Outlet_Size** is null/missing and **Outlet_Location_Type** is Tier 2, to impute those missing values.

# CELL ********************

grocery_store_indices = train_df[train_df['Outlet_Size'].isnull()].query(" Outlet_Type == 'Grocery Store' ").index

tier_2_indices = train_df[train_df['Outlet_Size'].isnull()].query(" Outlet_Location_Type == 'Tier 2' ").index

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, we are updating those indices for the column **Outlet_Size** with the value Small.

# CELL ********************

train_df.loc[grocery_store_indices, 'Outlet_Size'] = 'Small'

train_df.loc[tier_2_indices, 'Outlet_Size'] = 'Small'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Performing the same transformations on the **test** dataset.

# CELL ********************

grocery_store_indices = test_df[test_df['Outlet_Size'].isnull()].query(" Outlet_Type == 'Grocery Store' ").index

tier_2_indices = test_df[test_df['Outlet_Size'].isnull()].query(" Outlet_Location_Type == 'Tier 2' ").index

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

test_df.loc[grocery_store_indices, 'Outlet_Size'] = 'Small'

test_df.loc[tier_2_indices, 'Outlet_Size'] = 'Small'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# After we have imputed the missing values, let's check again if we still have missing values in both train and test datasets. 

# CELL ********************

train_df.isnull().sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

test_df.isnull().sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# There are **no missing values** in the datasets.

# MARKDOWN ********************

# ### **Feature Engineering**

# MARKDOWN ********************

# Now that we have completed the data understanding and data preparation step, before starting with the modeling task, we think that certain features are not present in the dataset, but we can create them using the existing columns, which can have the predictive power to predict the sales. This step of creating a new feature from the existing features in the dataset is known as **Feature Engineering**. So, we will start with a hypothesis - As the store gets older, the sales increase. Now, how do we define old? We know the establishment year and this data is collected in 2013, so the age can be found by subtracting the establishment year from 2013. This is what we are doing in the below code.

# MARKDOWN ********************

# We are creating a new feature **Outlet_Age** which indicates how old the outlet is.

# CELL ********************

train_df['Outlet_Age'] = 2013 - train_df['Outlet_Establishment_Year']

test_df['Outlet_Age'] = 2013 - test_df['Outlet_Establishment_Year']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

fig = plt.figure(figsize = (18, 6))

sns.boxplot(x = 'Outlet_Age', y = 'Item_Outlet_Sales', data = train_df);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# * The hypothesis that we had - **As the store gets older, the sales increase** does not seem to hold based on the above plot. Because of the different ages of stores, the sales have similar distribution approximately. But let's keep this variable as of now, and we will revisit this variable at the time of model building and we can remove this variable by observing its significance later on.

# MARKDOWN ********************

# ## **Modeling**

# MARKDOWN ********************

# Now, that we have analyzed all the variables in the dataset, we are ready to start building the model. We have observed that not all the independent variables are important to predict the outcome variable. But at the beginning, we will use all the variables, and then from the model summary, we will decide on which variable to remove from the model. Model building is an iterative task.

# CELL ********************

# We are removing the outcome variable from the feature set
# Also removing the variable Outlet_Establishment_Year, as we have created a new variable Outlet_Age
train_features = train_df.drop(['Item_Outlet_Sales', 'Outlet_Establishment_Year'], axis = 1)

# And then we are extracting the outcome variable separately
train_target = train_df['Item_Outlet_Sales']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Whenever we have categorical variables as independent variables, we need to create **one hot encoded** representation (which is also known as dummy variables) of those categorical variables. The below code is creating dummy variables and we are removing the first category in those variables, known as **reference variable**. The reference variable helps to interpret the linear regression which we will see later.

# CELL ********************

# Creating dummy variables for the categorical variables
train_features = pd.get_dummies(train_features, drop_first = True)

train_features.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - Notice that the column names of all the categorical variables, and also the overall number of columns has increased after we created the dummy variables.
# - For each of those categorical variables, the first category has been removed, e.g., the category **Low Fat** of the categorical variable **Item_Fat_Content** has been removed and became the **reference variable**, and we only have the category **Regular** as a new column **Item_Fat_Content_Regular**.

# MARKDOWN ********************

# Below, we are scaling the numerical variables in the dataset to have the same range. If we don't do this, then the model will be biased towards a variable where we have a higher range and the model will not learn from the variables with a lower range. There are many ways to do scaling. Here, we are using **MinMaxScaler** as we have both categorical and numerical variables in the dataset and don't want to change the dummy encodings of the categorical variables that we have already created. For more information on different ways of doing scaling, refer to the **section 6.3.1** of this page [here](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)

# CELL ********************

# Creating an instance of the MinMaxScaler
scaler = MinMaxScaler()

# Applying fit_transform on the training features data
train_features_scaled = scaler.fit_transform(train_features)

# The above scaler returns the data in array format, below we are converting it back to pandas DataFrame
train_features_scaled = pd.DataFrame(train_features_scaled, index = train_features.index, columns = train_features.columns)

train_features_scaled.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, as the dataset is ready, we are set to build the model using the **statsmodels** package.

# CELL ********************

# Adding the intercept term
train_features_scaled = sm.add_constant(train_features_scaled)

# Calling the OLS algorithm on the train features and the target variable
ols_model_0 = sm.OLS(train_target, train_features_scaled)

# Fitting the Model
ols_res_0 = ols_model_0.fit()

print(ols_res_0.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - We can see that the **R-squared** for the model is **0.563**. 
# - Not all the variables are statistically significant to predict the outcome variable. To check which variables are statistically significant or have predictive power to predict the target variable, we need to check the **p-value** against all the independent variables.
# 
# **Interpreting the Regression Results:**
# 
# 1. **Adj. R-squared**: It reflects the fit of the model.
#     - Adjusted R-squared values range from 0 to 1, where a higher value generally indicates a better fit, assuming certain conditions are met.
#     - In our case, the value for Adjusted R-squared is **0.562**.
# 
# 2. **coeff**: It represents the change in the output Y due to a change of one unit in the independent variable (everything else held constant).
# 3. **std err**: It reflects the level of accuracy of the coefficients.
#     - The lower it is, the more accurate the coefficients are.
# 4. **P >|t|**: It is the p-value.
#    
#    * Pr(>|t|) : For each independent feature, there is a null hypothesis and alternate hypothesis.
# 
#     Ho : Independent feature is not significant. 
#    
#     Ha : Independent feature is significant. 
#     
#    * The p-value of less than 0.05 is considered to be statistically significant with a confidence level of 95%. 
# 
#    
# 5. **Confidence Interval**: It represents the range in which our coefficients are likely to fall (with a likelihood of 95%).


# MARKDOWN ********************

# To understand in detail how p-values can help to identify statistically significant variables to predict the sales, we need to understand the hypothesis testing framework.

# MARKDOWN ********************

# In the following example, we are showing the null hypothesis between the independent variable **Outlet_Size** and the dependent variable **Item_Outlet_Sales** to identify if there is any relationship between them or not.

# MARKDOWN ********************

# - **Null hypothesis:** There is nothing going on or there is no relationship between variables **Outlet_Size** and **Item_Outlet_Sales**, i.e.

# MARKDOWN ********************

# <center>$\mu_{\text {outlet size }=\text { High }}=\mu_{\text {outlet size }=\text { Medium }}=\mu_{\text {outlet size }=\text { Small }}$</center> where, $\mu$ represents mean sales.

# MARKDOWN ********************

# - **Alternate hypothesis:** There is something going on, because of which, there is a relationship between variables **Outlet_Size** and **Item_Outlet_Sales**, i.e. 

# MARKDOWN ********************

# <center>$\mu_{\text {outlet size = High }} \neq \mu_{\text {outlet size = Medium }} \neq \mu_{\text {outlet size }=\text { small }}$</center> where, $\mu$ represents mean sales.

# MARKDOWN ********************

# From the above model summary, if the p-value is less than the significance level of 0.05, then we will reject the null hypothesis in favor of the alternate hypothesis. In other words, we have enough statistical evidence that there is some relationship between the variables **Outlet_Size** and **Item_Outlet_Sales**.

# MARKDOWN ********************

# Based on the above analysis, if we observe the above model summary, we can see only some of the variables or some of the categories of a categorical variables have a p-value lower than 0.05.

# MARKDOWN ********************

# ## **Feature Selection**

# MARKDOWN ********************

# ### **Removing Multicollinearity**


# MARKDOWN ********************

# * Multicollinearity occurs when predictor variables in a regression model are correlated. This correlation is a problem because predictor variables should be independent. If the correlation between independent variables is high, it can cause problems when we fit the model and interpret the results. When we have multicollinearity in the linear model, the coefficients that the model suggests are unreliable.
# 
# * There are different ways of detecting (or testing) multicollinearity. One such way is the Variation Inflation Factor.
# 
# * **Variance Inflation factor**:  Variance inflation factor measures the inflation in the variances of the regression parameter estimates due to collinearities that exist among the predictors.  It is a measure of how much the variance of the estimated regression coefficient βk is “inflated” by the existence of correlation among the predictor variables in the model. 
# 
# * General Rule of thumb: If VIF is 1, then there is no correlation between the kth predictor and the remaining predictor variables, and hence the variance of β̂k is not inflated at all. Whereas, if VIF exceeds 5 or is close to exceeding 5, we say there is moderate VIF and if it is 10 or exceeds 10, it shows signs of high multicollinearity.


# CELL ********************

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled.values, i) for i in range(train_features_scaled.shape[1])],
    index = train_features_scaled.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Outlet_Age has a high VIF score. Hence, we are dropping Outlet_Age and building the model.

# CELL ********************

train_features_scaled_new = train_features_scaled.drop("Outlet_Age", axis = 1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new.values, i) for i in range(train_features_scaled_new.shape[1])],
    index = train_features_scaled_new.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, let's build the model and observe the p-values.

# CELL ********************

ols_model_2 = sm.OLS(train_target, train_features_scaled_new)

ols_res_2 = ols_model_2.fit()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(ols_res_2.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# It is not a good practice to consider VIF values for dummy variables as they are correlated to other categories and hence have a high VIF usually. So, we built the model and calculated the p-values. We can see that all the categories in the Item_Type column show a p-value higher than 0.05. So, we can drop the Item_Type column.

# CELL ********************

train_features_scaled_new2 = train_features_scaled_new.drop(['Item_Type_Breads',
'Item_Type_Breakfast',
'Item_Type_Canned',
'Item_Type_Dairy',
'Item_Type_Frozen Foods',
'Item_Type_Fruits and Vegetables',
'Item_Type_Hard Drinks',
'Item_Type_Health and Hygiene',
'Item_Type_Household',
'Item_Type_Meat',
'Item_Type_Others',
'Item_Type_Seafood',
'Item_Type_Snack Foods',
'Item_Type_Soft Drinks',
'Item_Type_Starchy Foods'], axis = 1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new2.values, i) for i in range(train_features_scaled_new2.shape[1])],
    index = train_features_scaled_new2.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ols_model_3 = sm.OLS(train_target, train_features_scaled_new2)

ols_res_3 = ols_model_3.fit()

print(ols_res_3.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, from the p-values, we can see that we can remove the Item_Weight column as it has the highest p-value, i.e., it is the most insignificant variable.

# CELL ********************

train_features_scaled_new3 = train_features_scaled_new2.drop("Item_Weight", axis = 1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new3.values, i) for i in range(train_features_scaled_new3.shape[1])],
    index = train_features_scaled_new3.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Let's build the model again.

# CELL ********************

ols_model_4 = sm.OLS(train_target, train_features_scaled_new3)

ols_res_4 = ols_model_4.fit()

print(ols_res_4.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# From the above p-values, both the categories of the column Outlet_Location_Type have a p-value lower than 0.05. So, we can remove the categories of Outlet_Location_Type.

# CELL ********************

train_features_scaled_new4 = train_features_scaled_new3.drop(["Outlet_Location_Type_Tier 2", "Outlet_Location_Type_Tier 3"], axis = 1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new4.values, i) for i in range(train_features_scaled_new4.shape[1])],
    index = train_features_scaled_new4.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ols_model_5 = sm.OLS(train_target, train_features_scaled_new4)

ols_res_5 = ols_model_5.fit()

print(ols_res_5.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# From the above p-values, both the categories of the column Outlet_Size have a p-value lower than 0.05. So, we can remove the categories of Outlet_Size.

# CELL ********************

train_features_scaled_new5 = train_features_scaled_new4.drop(["Outlet_Size_Small", "Outlet_Size_Medium"], axis=1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new5.values, i) for i in range(train_features_scaled_new5.shape[1])],
    index = train_features_scaled_new5.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ols_model_6 = sm.OLS(train_target, train_features_scaled_new5)
ols_res_6 = ols_model_6.fit()
print(ols_res_6.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Finally, let's drop the Item_Visibility.

# CELL ********************

train_features_scaled_new6 = train_features_scaled_new5.drop("Item_Visibility", axis = 1)

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new6.values, i) for i in range(train_features_scaled_new6.shape[1])],
    index = train_features_scaled_new6.columns,
    dtype = float)

print("VIF Scores: \n\n{}\n".format(vif_series))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ols_model_7 = sm.OLS(train_target, train_features_scaled_new6)
ols_res_7 = ols_model_7.fit()
print(ols_res_7.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - All the VIF Scores are now less than 5 indicating no multicollinearity.
# - Now, all the p values are lesser than 0.05 implying all the current variables are significant for the model.
# - The R-Squared value did not change by much. It is still coming out to be ~0.56 which implies that all other variables were not adding any value to the model.
# 
# Lets' check the assumptions of the linear regression model.

# MARKDOWN ********************

# ## **Checking for the assumptions and rebuilding the model**

# MARKDOWN ********************

# In this step, we will check whether the below assumptions hold true or not for the model. In case there is an issue, we will rebuild the model after fixing those issues.

# MARKDOWN ********************

# 1. Mean of residuals should be 0
# 2. Normality of error terms
# 3. Linearity of variables
# 4. No heteroscedasticity

# MARKDOWN ********************

# ### **Mean of residuals should be 0 and normality of error terms**

# CELL ********************

# Residuals
residual = ols_res_7.resid 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

residual.mean()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - The mean of residuals is very close to 0. Hence, the corresponding assumption is satisfied.

# MARKDOWN ********************

# ### **Tests for Normality**
# 
# **What is the test?**
# 
# * Error terms/Residuals should be normally distributed.
# 
# * If the error terms are non-normally distributed, confidence intervals may become too wide or narrow. Once the confidence interval becomes unstable, it leads to difficulty in estimating coefficients based on the minimization of least squares.
# 
# **What does non-normality indicate?**
# 
# * It suggests that there are a few unusual data points that must be studied closely to make a better model.
# 
# **How to check the normality?**
# 
# * We can plot the histogram of residuals and check the distribution visually.
# 
# * It can be checked via QQ Plot. Residuals following normal distribution will make a straight line plot otherwise not.
# 
# * Another test to check for normality: The Shapiro-Wilk test.
# 
# **What if the residuals are not-normal?**
# 
# * We can apply transformations like log, exponential, arcsinh, etc. as per our data.

# CELL ********************

# Plot histogram of residuals
sns.histplot(residual, kde = True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# We can see that the error terms are normally distributed. The assumption of normality is satisfied.

# MARKDOWN ********************

# ### **Linearity of Variables**
# 
# It states that the predictor variables must have a linear relation with the dependent variable.
# 
# To test this assumption, we'll plot the residuals and the fitted values and ensure that residuals do not form a strong pattern. They should be randomly and uniformly scattered on the x-axis.

# CELL ********************

# Predicted values
fitted = ols_res_7.fittedvalues

sns.residplot(x = fitted, y = residual, color = "lightblue")

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - We can see that there is some pattern in fitted values and residuals, i.e., the residuals are not randomly distributed.
# - Let's try to fix this. We can apply the log transformation on the target variable and try to build a new model.

# CELL ********************

# Log transformation on the target variable
train_target_log = np.log(train_target)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fitting new model with the transformed target variable
ols_model_7 = sm.OLS(train_target_log, train_features_scaled_new6)

ols_res_7 = ols_model_7.fit()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Predicted values
fitted = ols_res_7.fittedvalues

residual1 = ols_res_7.resid

sns.residplot(x = fitted, y = residual1, color = "lightblue")

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - We can see that there is no pattern in the residuals vs fitted values scatter plot now, i.e., the linearity assumption is satisfied.
# - Let's check the model summary of the latest model we have fit. 

# CELL ********************

print(ols_res_7.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - The model performance has improved significantly. The R-Squared has increased from 0.56 to 0.720.
# 
# Let's check the final assumption of homoscedasticity.

# MARKDOWN ********************

# ### **No Heteroscedasticity**

# MARKDOWN ********************

# #### **Test for Homoscedasticity**
# 
# * **Homoscedasticity -** If the variance of the residuals are symmetrically distributed across the regression line, then the data is said to homoscedastic.
# 
# * **Heteroscedasticity -** If the variance is unequal for the residuals across the regression line, then the data is said to be heteroscedastic. In this case, the residuals can form an arrow shape or any other non symmetrical shape.
# 
# - We will use Goldfeld–Quandt test to check homoscedasticity.
# 
#     - Null hypothesis : Residuals are homoscedastic
# 
#     - Alternate hypothesis : Residuals are hetroscedastic

# CELL ********************

from statsmodels.stats.diagnostic import het_white

from statsmodels.compat import lzip

import statsmodels.stats.api as sms

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ["F statistic", "p-value"]

test = sms.het_goldfeldquandt(train_target_log, train_features_scaled_new6)

lzip(name, test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - As we observe from the above test, the p-value is greater than 0.05, so we fail to reject the null-hypothesis. That means the residuals are homoscedastic.

# MARKDOWN ********************

# We have verified all the assumptions of the linear regression model. The final equation of the model is as follows:

# MARKDOWN ********************

# **$\log ($ Item_Outlet_Sales $)$ $= 4.6356 + 1.9555 *$ Item_MRP$ + 0.0158 *$ Item_Fat_Content_Regular $ + 1.9550 *$ Outlet_Type_Supermarket Type1 $ + 1.7737 *$ Outlet_Type_Supermarket Type2$ + 2.4837*$ Outlet_Type_Supermarket Type3**

# MARKDOWN ********************

# Now, let's make the final test predictions.

# CELL ********************

without_const = train_features_scaled.iloc[:, 1:]

test_features = pd.get_dummies(test_df, drop_first = True)

test_features = test_features[list(without_const)]

# Applying transform on the test data
test_features_scaled = scaler.transform(test_features)

test_features_scaled = pd.DataFrame(test_features_scaled, columns = without_const.columns)

test_features_scaled = sm.add_constant(test_features_scaled)

test_features_scaled = test_features_scaled.drop(["Item_Weight", "Item_Visibility", "Item_Type_Breads", "Item_Type_Breakfast", "Item_Type_Canned", "Item_Type_Dairy","Item_Type_Frozen Foods","Item_Type_Fruits and Vegetables", "Item_Type_Hard Drinks", "Item_Type_Health and Hygiene", "Item_Type_Household", "Item_Type_Meat", "Item_Type_Others", "Item_Type_Seafood", "Item_Type_Snack Foods", "Item_Type_Soft Drinks", "Item_Type_Starchy Foods", "Outlet_Size_Medium", "Outlet_Size_Small", "Outlet_Location_Type_Tier 2", "Outlet_Location_Type_Tier 3", 'Outlet_Age'], axis = 1)

test_features_scaled.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## **Evaluation Metrics**

# MARKDOWN ********************

# ### **R-Squared**

# MARKDOWN ********************

# The R-squared metric gives us an indication that how good/bad our model is from a baseline model. Here, we have explained ~98% variance in the data as compared to the baseline model when there is no independent variable.

# CELL ********************

print(ols_res_7.rsquared)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### **Mean Squared Error**

# MARKDOWN ********************

# This metric measures the average of the squares of the errors, i.e., the average squared difference between the estimated values and the actual value.

# CELL ********************

print(ols_res_7.mse_resid)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### **Root Mean Squared Error**

# MARKDOWN ********************

# This metric is the same as the above but, instead of simply taking the average, we also take the square root of MSE to get RMSE. This helps to get the metric in the same unit as the target variable.

# CELL ********************

print(np.sqrt(ols_res_7.mse_resid))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Below, we are checking the cross-validation score to identify if the model that we have built is **underfitted**, **overfitted** or **just right fit** model.

# CELL ********************

# Fitting linear model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linearregression = LinearRegression()                                    

cv_Score11 = cross_val_score(linearregression, train_features_scaled_new6, train_target_log, cv = 10)

cv_Score12 = cross_val_score(linearregression, train_features_scaled_new6, train_target_log, cv = 10, 
                             scoring = 'neg_mean_squared_error')                                  


print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std()*2))

print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std()*2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Observations:**
# 
# - The R-Squared on the cross-validation is 0.718 which is almost similar to the R-Squared on the training dataset.
# - The MSE on cross-validation is 0.290 which is almost similar to the R-Squared on the training dataset.

# MARKDOWN ********************

# It seems like that our model is **just right fit**. It is giving a generalized performance.

# MARKDOWN ********************

# Since this model that we have developed is a linear model which is not capable of capturing non-linear patterns in the data, we may want to build more advanced regression model which can capture the non-linearities in the data and improve this model further. But that is out of the scope of this case study.

# MARKDOWN ********************

# ## **Predictions on the Test Dataset**

# MARKDOWN ********************

# Once our model is built and validated, we can now use this to predict the sales in our test data as shown below:

# CELL ********************

# These test predictions will be on a log scale
test_predictions = ols_res_7.predict(test_features_scaled)

# We are converting the log scale predictions to its original scale
test_predictions_inverse_transformed = np.exp(test_predictions)

test_predictions_inverse_transformed

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Point to remember:** The output of this model is in log scale. So, after making prediction, we need to transform this value from log scale back to its original scale by doing the inverse of log transformation, i.e., taking exponentiation.</font>

# CELL ********************

fig, ax = plt.subplots(1, 2, figsize = (24, 12))

sns.histplot(test_predictions, ax = ax[0]);

sns.histplot(test_predictions_inverse_transformed, ax = ax[1]);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## **Conclusions and Recommendations**

# MARKDOWN ********************

# - We performed EDA, univariate and bivariate analysis, on all the variables in the dataset.
# - We then performed missing values treatment using the relationship between variables.
# - We started the model building process with all the features.
# - We removed multicollinearity from the data and analyzed the model summary report to drop insignificant features.
# - We checked for different assumptions of linear regression and fixed the model iteratively if any assumptions did not hold true.
# - Finally, we evaluated the model using different evaluation metrics.

# MARKDOWN ********************

# Lastly below is the model equation:

# MARKDOWN ********************

# **$\log ($ Item_Outlet_Sales $)$ $= 4.6356 + 1.9555 *$ Item_MRP$ + 0.0158 *$ Item_Fat_Content_Regular $ + 1.9550 *$ Outlet_Type_Supermarket Type1 $ + 1.7737 *$ Outlet_Type_Supermarket Type2$ + 2.4837*$ Outlet_Type_Supermarket Type3**

# MARKDOWN ********************

# - From the above equation, we can interpret that, with one unit change in the variable **Item_MRP**, the outcome variable, i.e., log of **Item_Outlet_Sales** increases by 1.9555 units. So, if we want to increase the sales, we may want to store higher MRP items in the high visibility area.
# 
# - On average, keeping other factors constant, the log sales of stores with type Supermarket type 3 are 1.4 (2.4837/1.7737) times the log sales of stores with type 2 and 1.27 (2.4837/1.9550) times the log sales of those stores with type 1.

# MARKDOWN ********************

# After interpreting this linear regression equation, it is clear that large stores of supermarket type 3 have more sales than other types of outlets. So, we want to maintain or improve the sales in these stores and for the remaining ones, we may want to make strategies to improve the sales, for example, providing better customer service, better training for store staffs, providing more visibility of high MRP item, etc.
