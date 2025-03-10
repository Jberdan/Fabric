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
# META       "default_lakehouse_workspace_id": "a601866f-f379-44b7-b085-0e5d028917d2",
# META       "known_lakehouses": [
# META         {
# META           "id": "296e29e0-1c40-4d91-b5b6-89e6736c6191"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Load data using notebooks

# CELL ********************

from pyspark.sql import Row

Customer = Row("FirstName","LastName","Email","LoyaltyPoints")

customer_1 = Customer('John', 'Smith', 'john.smith@contoso.com', 15)
customer_2 = Customer('Anna', 'Miller', 'anna.miller@contoso.com', 65)
customer_3 = Customer('Sam', 'Walters', 'sam@contoso.com', 6)
customer_4 = Customer('Mark', 'Duffy', 'mark@contoso.com', 78)

customers = [customer_1, customer_2, customer_3, customer_4]
df= spark.createDataFrame(customers)

df.write.parquet("Files/customers")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.read.parquet("Files/customers")

display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
