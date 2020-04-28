import numpy as np
import pandas as pd
import os


def load_training_data(data_dir='data', data_file='training_data_standardized.csv'):
    """Loads the training data from csv file."""

    file = os.path.join(data_dir, data_file)
    try:
        with open(file, "rb") as f:
            training_data_df = pd.read_csv(f, index_col=0)
        print("Read training data from", file)
    except:
        print("Unable to read file")

    return training_data_df

