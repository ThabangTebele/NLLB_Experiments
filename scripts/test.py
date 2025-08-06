import pandas as pd

test_df = pd.read_csv("data/combined_dataset.csv")
print(test_df.columns.tolist())
print(test_df.head())
