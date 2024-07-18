import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model,find_best_model
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

@dataclass
class ModelTrainerConfig():
    preprocessing_obj_filepath: str = os.path.join('artifact', 'preprocessing.pkl')
    model_filepath: str = os.path.join('artifact','model.pkl')

class ModelTrainer():
    def __init__(self) -> None:
        self.config = ModelTrainerConfig()
    
    def initiate_model_training(self,X_train,y_train,X_test,y_test):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "KNN Regression": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "xgboost": XGBRegressor(),
                #"catboost": CatBoostRegressor(verbose=False),
                "LightBGM": LGBMRegressor()
            }
            evaluated_models = evaluate_model(models,X_train,y_train,X_test,y_test)
            best_model_name,best_model_obj,best_best_score = find_best_model(evaluated_models)
            logging.info(f"Successful found best model: {best_model_name} with score{best_best_score}")
            save_object(best_model_obj,self.config.model_filepath)
        except Exception as e:
            logging.error(f"Error Occured in Model Training: {e}")
            raise CustomException(e,sys)

        
