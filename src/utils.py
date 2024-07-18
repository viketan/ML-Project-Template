import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from src.constant import hyperparams
from sklearn.model_selection import GridSearchCV

def save_object(obj,filepath):
    try:
        with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        logging.info(f"Successfully Saved Object to {filepath}")
    except Exception as e:
         logging.error(f"Error occured in saving object: {e}")
         raise CustomException(e,sys)

def evaluate_model(models,X_train,y_train,X_test,y_test):
    try:
        evaluated_model = {}
        for key, model in models.items():
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test,y_pred)
            evaluated_model[key] = [model,score]
        logging.info(f"Successfully evaluate all models: {evaluated_model}")
        return evaluated_model
    except Exception as e:
         logging.error(f"Error Occured in evaluting models: {e}")
         raise CustomException(e,sys)

def find_best_model(evaluated_models):
    try:
        best_model_key = None
        best_model_obj = None
        best_score = -float('inf')
        
        for key, (model, score) in evaluated_models.items():
            if score > best_score:
                best_score = score
                best_model_obj = model
                best_model_key = key
        
        best_model = evaluated_models[best_model_key]
        return best_model_key, best_model_obj, best_score
    except Exception as e:
        logging.error(f"Error occurred in finding the best model: {e}")
        raise CustomException(e, sys)

def hyperparameter_tuning(best_model_name, best_model, X_train, y_train):
        try:
            if best_model_name in hyperparams:
                param_grid = hyperparams[best_model_name]
                grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, scoring='r2', cv=5)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                logging.info(f"Hyperparameter tuning completed for {best_model_name}. Best parameters: {best_params}")
            else:
                logging.warning(f"No hyperparameters defined for {best_model_name}. Skipping tuning.")
        except Exception as e:
            logging.error(f"Error occurred during hyperparameter tuning for {best_model_name}: {e}")
            raise CustomException(e, sys)
        return best_model

          