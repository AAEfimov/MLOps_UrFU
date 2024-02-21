from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
import numpy as np 
import pickle


from config import *
from aux_func import *

def get_model_or_pipeline():
    """
    Change this function to switch models
    """
    return LinearRegression()


X_train, y_train = load_datasets('train/X_train.npy', 'train/y_train.npy')


reg_model = get_model_or_pipeline()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_train)

print(f"Train error: ", np.sqrt(mean_squared_error(y_train, y_pred)))

save_model(model_file, reg_model)

