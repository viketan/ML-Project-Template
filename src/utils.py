import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score

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


          