from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
import numpy as np 

import pickle
from config import *

# load train DATA
X_train = np.load('train/X_train.npy', allow_pickle=True)
y_train = np.load('train/y_train.npy', allow_pickle=True)

# Train model
reg_model = LinearRegression().fit(X_train, y_train)

y_pred = reg_model.predict(X_train)

print(f"Train score: ", np.sqrt(mean_squared_error(y_train, y_pred)))

# save model
with open(model_file,'wb') as f:
    pickle.dump(reg_model,f)

