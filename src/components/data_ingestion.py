import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from ..exception import CustomException
from ..logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated Data Ingestion")
        try:
            # Read data from source
            df = pd.read_csv("../data/stud.csv")
            logging.info("Successfully read data from source")

            # Create directory for artifacts if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data")

            # Split data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=123)

            # Save training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Saved train and test data")
            logging.info("Successfully completed data ingestion process")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)
