"""
model_preparation.py
Fit model or load pretrainded.
"""

__author__ = "UrFU team"
__copyright__ = "Copyright 2023, Planet Earth"

import os
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from aux_func import load_datasets, save_model
from config import model_file


def get_model_or_pipeline(use_pretrained = False):
    """
    Change this function to switch models
    """
    if use_pretrained:
        with open(model_file, "rb") as f:
            reg_model = pickle.load(f)
            return reg_model
    else:
        reg_model = RandomForestRegressor(n_estimators=1000, max_depth=7).fit(X_train, y_train)
        return reg_model

if __name__ == '__main__':

    use_pretrained = False

    X_train, y_train = load_datasets("train/X_train.npy", "train/y_train.npy")

    if os.path.exists(model_file):
        use_pretrained = True
        print("Use pretrained model")

    reg_model = get_model_or_pipeline(use_pretrained)

    y_pred = reg_model.predict(X_train)

    print("Train mse: ", mean_squared_error(y_train, y_pred))
    print("Train mae: ",  mean_absolute_error(y_train, y_pred))
    print("Train r^2: ",  r2_score(y_train, y_pred))

    if use_pretrained == False:
        save_model(model_file, reg_model)
