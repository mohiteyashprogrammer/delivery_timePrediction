import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transdormation Initated")

            numerical_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Weather_conditions',
            'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
            'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City']

            logging.info("Pipline initiated")


            # Num_pipline
            num_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

            cato_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("scaler",StandardScaler())
        ]
    )

            # Create Preprocessor object
            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features)
            ])

            return preprocessor

            logging.info("Pipline Complited")

        
        except Exception as e:
            logging.info("Error Occured In Data Ingeation Class")
            raise CustomException(e, sys)




    def initated_data_transformation(self,train_path,test_path):
        try:
            train_data  = pd.read_csv(train_path)
            test_data  = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')

            logging.info("Optaning Preprocessor Object")

            preprocessor_obj = self.get_data_transformation_obj()

            target_colum_name = "Time_taken (min)"
            drop_column = [target_colum_name]

            ## Saprate Indipendent and Dependent Features like x and y
            input_features_train_data = train_data.drop(drop_column,axis=1)
            target_feature_train_data = train_data[target_colum_name]

            input_features_test_data = test_data.drop(drop_column,axis=1)
            target_feature_test_data = test_data[target_colum_name]


            ## Apply Transformation using Preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("Application Preprocessor on Train and Test data")

            ## Convert in to Array To be fast
            train_array = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]


            ## Calling Save object Function and save preprocessor pickel file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor_obj)

            logging.info("Saving Preprocessor Pikel File")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error Occured In Data Transformation Stage")
            raise CustomException(e, sys)


