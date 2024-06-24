import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def save_obj(file_path, obj):
    """
    Save a Python object to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"An error occurred while saving object to {file_path}: {e}")
        raise Exception(e, sys)

def load_obj(file_path):
    """
    Load a Python object from a file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        print(f"An error occurred while loading object from {file_path}: {e}")
        raise Exception(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,parameters):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=parameters[list(models.keys())[i]]
            gs=GridSearchCV(model,para,cv=3,n_jobs=-1)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            y_train_pred=model.predict(X_train)
            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
            


    except Exception as e:
        raise Exception(e,sys)