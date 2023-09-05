import os
import sys
from dataclasses import dataclass
import xgboost
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def Initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:-1],

            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K- Nearest Neighbour":KNeighborsRegressor(),
                "XGB regressor":XGBRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
            }
            params={
                "Decision tree":{'criterion':['squared_error','friedman_mse','absolute_error','poisson']},
                "Random forest":{'n_estimators':[8.16,32,64,128,256]},
                "Gradient Boosting":{'learning_rate':[0.1,0.01,0.05,0.001],'subsample':[0.6,0.7,0.75,0.8,0.85,0.9], 'n_estimators':[8,16,32,64,128,256]},
                "KNN":{"n_neighbour":[5,7,9,11]},
                "XGBRegressor":{
                    'learning rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Adaboost":{
                    'learning_rate':[0.1,0.01,0.5,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                }


            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models, params=params)

            ## to get best model score 
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square= r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
        