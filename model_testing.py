
from sklearn import datasets
import pickle


# load
with open('model.pkl', 'rb') as f:
    reg_model = pickle.load(f)



y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

reg_model.score(X_test, y_test)