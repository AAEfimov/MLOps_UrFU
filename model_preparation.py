
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
import pickle


# load train datasets from files


# train linear regression model

reg_model = LinearRegression().fit(X_train, y_train)

y_pred = reg_model.predict(X_train)

print(" train mean_squared_error: ", np.sqrt(mean_squared_error(y_train, y_pred)))

# save model using pickle

with open('model.pkl','wb') as f:
    pickle.dump(reg_model,f)
