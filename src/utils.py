import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle

def save_object(obj,filepath):
    try:
        with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        logging.info(f"Successfully Saved Object to {filepath}")
    except Exception as e:
         logging.error(f"Error occured in saving object: {e}")
         raise CustomException(e,sys)
    