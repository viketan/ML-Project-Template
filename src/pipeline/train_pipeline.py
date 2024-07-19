import os
import sys
import warnings
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

warnings.filterwarnings('ignore')  # Suppressing warnings

class Train:
    def __init__(self):
        try:
            logging.info("Initiating ML project")
            
            # Data Ingestion
            data_ingestion = DataIngestion() 
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully")
            
            # Data Transformation
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
            logging.info("Data transformation completed successfully")
            
            # Model Training
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)
            logging.info("Model training completed successfully")
            
        except Exception as e:
            logging.error(f"Error occurred in Train class: {e}")
            raise CustomException(e, sys)

