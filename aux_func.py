from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
import numpy as np
import pickle


def load_datasets(X_train_path, y_train_path):
    """
    Function to open the train dataset and answers
    """
    X_train = np.load(X_train_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)

    return X_train, y_train

def save_model(path, model):
    """
    Save trainde model to pkl file
    """
    with open(path,'wb') as f:
        pickle.dump(model,f)

