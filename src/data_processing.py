import pandas as pd
import json
from sklearn.model_selection import train_test_split


def csv_data_split(csv_path, test_size=0.20, random_state=42):
    df = pd.read_csv(csv_path)

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state,stratify=df["Class"])
    sample_df = train_data.sample(n=int(len(train_data) / 2), random_state=random_state)
    return train_data, test_data, sample_df


def metadata(file_path):
    with open(file_path) as f:
        my_metadata_dict = json.load(f)
    return my_metadata_dict
