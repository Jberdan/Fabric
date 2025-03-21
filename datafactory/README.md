# 1. End-to-End data Integration (Ingest and Transform)

## Use cases:
a. Ingest data using the copy assistant in a pipeline
  - blob storage table in a data lakehouse
b. Transform the data using a dataflow
  - process the raw data and move it to the same data lakehouse

Sample dataset NYC-Taxi

NOTE: Workspace created has Git Integration

In this project, I utilized "Copy data assistant" for copy to activity.  At the "Choose data source", I moved my cursor in the abric menu tab and clicked on "Sample data" tab then choose NYC Taxi - Green then system will then move to next stage "Connect to data source" and preview the data. Once confirmed, I hit "Next" to choose data destination and selected lakehouse and then "Next".  Since I haven't created a lakehouse at this point, I opted to clicked the radio button "Create new lakehouse" and entered lakeshoue name, in this case "lh-nyctaxi" then clicked "Next". Then I configured the details of my lakehsoue destination, in this case I wanted to store the data into a table and named "
