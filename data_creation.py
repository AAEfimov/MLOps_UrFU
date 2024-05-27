"""
data_creation.py
Tests for data on pretrained model.
"""

__author__ = "UrFU team"
__copyright__ = "Copyright 2023, Planet Earth"

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import prepared_path


def save_datasets(X_train, X_test, y_train, y_test):
    dir_list = ["train", "test"]
    for ext_dir in dir_list:
        if not os.path.exists(ext_dir):
            os.mkdir(ext_dir)

    np.save("train/X_train.npy", X_train)
    np.save("train/y_train.npy", y_train)
    np.save("test/X_test.npy", X_test)
    np.save("test/y_test.npy", y_test)


if __name__ == '__main__':

    df = pd.read_csv(prepared_path)

    X = df.drop("PRICE", axis=1).values
    y = df["PRICE"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    save_datasets(X_train, X_test, y_train, y_test)
