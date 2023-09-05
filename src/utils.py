from ast import Param
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.components.Model_trainer import ModelTrainer, ModelTrainerConfig
from sklearn.model_selection import GridSearchCV

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error message occured in python script name [{0}], line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))

    return error_message
        

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super.__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message


def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e :
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range (len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs= GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            # model.fit(x_train,y_train)
            y_train_Pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train,y_train_Pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

        