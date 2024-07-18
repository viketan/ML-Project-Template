import os,sys
from src.logger import logging
from src.exception import CustomException
import warnings
warnings.filterwarnings('ignore')

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        logging.info("Inititaing ML project")
        data_injection = DataIngestion() 
        train_data,test_data = data_injection.initiate_data_ingestion()
        data_transformation = DataTransformation()
        X_train,y_train,X_test,y_test = data_transformation.initiate_data_transformation(train_data,test_data)
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(X_train,y_train,X_test,y_test)


    except Exception as e:
        logging.error("Error ocuured in main: {e}")
        raise CustomException(e,sys)

if __name__=="__main__":
    main()
