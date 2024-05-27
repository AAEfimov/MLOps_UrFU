"""
test_data.py
Tests for data on pretrained model.
"""

__author__ = "UrFU team"
__copyright__ = "Copyright 2023, Planet Earth"

import unittest
import pickle

from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import PredictionErrorDisplay
import numpy as np
from config import model_file
import aux_func

class TestMethods(unittest.TestCase):
    
    def calculate_metric(self, model_pipe, X, y, metric = r2_score):
        """Расчет метрики.
        Параметры:
        ===========
        model_pipe: модель или pipeline
        X: признаки
        y: истинные значения
        metric: метрика (r2 - по умолчанию)
        """
        y_model = model_pipe.predict(X)
        return metric(y, y_model)

    def check_r2(self, score):
        self.assertTrue(score > 0.6)

    def check_mae(self, score):
        self.assertTrue(score < 0.0006)
    
    def test_r2_train(self):

        with open(model_file, 'rb') as f:
            LR = pickle.load(f)

        X_train, y_train = aux_func.load_datasets("train/X_train.npy", "train/y_train.npy")
        r2_train = self.calculate_metric(LR, X_train, y_train)

        # print(f"test main R2 {r2_train}")
        
        self.check_r2(r2_train)

    def test_mae_train(self):
        """
        """
        with open(model_file, 'rb') as f:
            LR = pickle.load(f)
            
        X_train, y_train = aux_func.load_datasets("train/X_train.npy", "train/y_train.npy")

        mae_test = self.calculate_metric(LR, X_train, y_train, mae)
        # print(f"test main mae {mae_test}")
        
        self.check_mae(mae_test)


    def test_r2_ds1(self):

        with open(model_file, 'rb') as f:
            LR = pickle.load(f)

        X_test, y_test = aux_func.load_datasets("test/X_test.npy", "test/y_test.npy")

        r2_train = self.calculate_metric(LR, X_test, y_test)

        # print(f"test main R2 {r2_train}")

        self.check_r2(r2_train)

    def test_mae_ds1(self):
        """
        """
        with open(model_file, 'rb') as f:
            LR = pickle.load(f)

        X_test, y_test = aux_func.load_datasets("test/X_test.npy", "test/y_test.npy")

        mae_test = self.calculate_metric(LR, X_test, y_test, mae)
        # print(f"test main mae {mae_test}")

        self.check_mae(mae_test)

