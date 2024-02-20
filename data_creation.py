import os
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


dataset = 'https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market'
od.download(dataset) # insert ypu kaggle  username and key

path = "new-york-housing-market/NY-House-Dataset.csv"
df = pd.read_csv(path)

def save_datasets(X_train, X_test, y_train, y_test):
    dir_list = ["train", "test"]
    for ext_dir in dir_list:
        if not os.path.exists(ext_dir):
            os.mkdir(ext_dir)

X = df.drop("PRICE", axis=1).values
y = df["PRICE"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
save_datasets(X_train, X_test, y_train, y_test)
