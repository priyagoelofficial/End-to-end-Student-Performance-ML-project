import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
import os
from src.utils import save_object
from src.exception import CustomException
# from data_transformation import DataTransformation
# from data_transformation import DataTransformationconfig
from src.components.data_ingestion import DataIngestion
import dill


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender",'race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder),
                    ('scaler', StandardScaler())

                ]
            )

            logging.info("Numerical columns standard scaler completed")
            logging.info("Categorical columns one hot encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ('cat_pipeline', cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data copleted")
            logging.info("Obtaining preprocesssing object")

            preprocesssing_obj=self.get_data_transformer_object()

            target_columns_name="math score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_columns_name], axis=1)
            target_feature_train_df=train_df[target_columns_name]

            input_feature_test_df=test_df.drop(columns=[target_columns_name], axis=1)
            target_feature_test_df=test_df[target_columns_name]

            logging.info("Applying preprocessing on test and train dataframe")

            input_feature_train_arr=preprocesssing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocesssing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.INFO("Saved preprocessing onject")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocesssing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)


