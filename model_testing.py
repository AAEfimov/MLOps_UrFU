"""
model_testing.py
Simple test for model.
"""

__author__ = "UrFU team"
__copyright__ = "Copyright 2023, Planet Earth"

import pickle

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from aux_func import load_datasets
from config import model_file



if __name__ == '__main__':

    # load TEST DATA
    X_test, y_test = load_datasets("test/X_test.npy", "test/y_test.npy")


    with open(model_file, "rb") as f:
        reg_model = pickle.load(f)

    y_pred = reg_model.predict(X_test)

    print("Test mse: ", mean_squared_error(y_test, y_pred))
    print("Test mae: ", mean_absolute_error(y_test, y_pred))
    print("Test r^2: ", r2_score(y_test, y_pred))

