import pandas as pd

# Load the datasets
dataset_path = "D:/Capstone Project/sahayak-main/sahayak-main/flask/datasets/dataset.csv"
training_path = "D:/Capstone Project/sahayak-main/sahayak-main/flask/datasets/Training.csv"

dataset_df = pd.read_csv(dataset_path)
training_df = pd.read_csv(training_path)

# Get the columns
dataset_columns = set(dataset_df.columns)
training_columns = set(training_df.columns)

# Find the common columns
common_columns = dataset_columns.intersection(training_columns)

print("Common columns:", common_columns)

# Create a new DataFrame with only the common columns
new_training_df = dataset_df[common_columns]

# Ensure the columns are in the same order as in the original Training.csv
new_training_df = new_training_df[training_df.columns]

# Save the new Training.csv
new_training_path = "D:/Capstone Project/sahayak-main/sahayak-main/flask/datasets/Training_converted.csv"
new_training_df.to_csv(new_training_path, index=False)

print("New Training.csv created with common columns.")
