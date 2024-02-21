from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np # linear algebra
import pickle
from config import *

# load TEST DATA
X_test = np.load('test/X_test.npy', allow_pickle=True)
y_test = np.load("test/y_test.npy", allow_pickle=True)

# load model
with open(model_file, 'rb') as f:
    reg_model = pickle.load(f)

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


print(f"Model score {reg_model.score(X_test, y_test)}")
