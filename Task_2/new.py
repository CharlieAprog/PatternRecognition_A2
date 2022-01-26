import pandas as pd
import numpy as np

def read_data(file_path):
    df = pd.from_csv(file_path)
    df.drop(['Time', 'Amount'], 1)
    return df

def get_of_items_per_class(df):
    np.uniqu