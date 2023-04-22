import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,precision_score,f1_score,recall_score
from sklearn.ensemble import RandomForestRegressor

from src.utils import model_eveluation
from src.utils import save_object


@dataclass
class ModelTraningConfig:
    traning_model_file_path = os.path.join("artifcats","model.pkl")



class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()


    def initated_model_traning(self,train_array,test_array):
        try:
            logging.info("Saprate Dependent and Independent Features in train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Multipal Model Traning
            models =  {
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet(),
            "RandomForestRegressor":RandomForestRegressor(random_state=3)
        }

            model_report:dict = model_eveluation(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n***************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## To Get Best Model Score From Dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name:{best_model_name}, R2 Score:{best_model_score}")
            print("\n***************************************************************\n")
            logging.info(f"Best Model Found, Model Name:{best_model_name}, R2 Score:{best_model_score}")

            save_object(file_path=self.model_traner_config.traning_model_file_path
            , obj=best_model
            )

        except Exception as e:
            logging.info("Error Occured In Model TRaning")
            raise CustomException(e, sys)
            


