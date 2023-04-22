import os
import sys
import pickle
import numpy as np
import pandas as pd  
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



## This Function is Save Pickel file

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)




def model_eveluation(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model  = list(models.values())[i]
            # Train Model
            model.fit(X_train,y_train)
            # Make Pradiction
            y_test_pred = model.predict(X_test)

            # Get R2 Scores 
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)


# To Lode Pikel File
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error Occured In Load Pickel File from utils")
        raise CustomException(e, sys)

