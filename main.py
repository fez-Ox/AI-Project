import pandas as pd

train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")
dev_df = pd.read_csv("data/raw/dev.csv")

print(train_df.shape)
print(train_df.columns)
print(train_df.head())

# Number of unique articles
print(len(train_df["article"].unique()))
