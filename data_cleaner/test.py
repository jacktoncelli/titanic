import pandas as pd
import os

data_raw_dir = os.path.join(os.getcwd(), "data/raw")

test_csv = os.path.join(data_raw_dir, "test.csv")
train_csv = os.path.join(data_raw_dir, "train.csv")

test_df = pd.read_csv(test_csv)
train_df = pd.read_csv(train_csv)
