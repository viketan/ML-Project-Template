import os
import sys
import pandas as pd
import pickle
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..exception import CustomException
from ..logger import logging
from ..utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_obj_filepath: str = os.path.join('artifact', 'preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformation_object(self):
        logging.info("Entered Data Transformation component")
        try:
            # Define numerical and categorical columns
            num_cols = ["reading_score", "writing_score"]  
            cat_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]  

            logging.info("Creating numerical pipeline")
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            logging.info("Creating categorical pipeline")
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Combining numerical and categorical pipelines")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )

            logging.info("Data Transformation component created successfully")
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred in getting preprocessing object: {e}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_file_path, test_file_path):
        logging.info("Initiating data transformation process")
        try:
            preprocessor = self.get_data_transformation_object()
            
            logging.info("Reading training data")
            train_df = pd.read_csv(train_file_path)
            logging.info("Reading test data")
            test_df = pd.read_csv(test_file_path)
            
            target_col = 'math_score'
            logging.info("Splitting training data into features and target")
            X_train = train_df.drop(target_col, axis=1)
            y_train = train_df[target_col]
            
            logging.info("Splitting test data into features and target")
            X_test = test_df.drop(target_col, axis=1)
            y_test = test_df[target_col]

            logging.info("Transforming training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info("Transforming test data")
            X_test_transformed = preprocessor.transform(X_test)
            
            logging.info("Saving preprocessor object")
            save_object(preprocessor,self.config.preprocessing_obj_filepath)
                
            logging.info("Data transformation process completed successfully")
            return X_train_transformed, y_train.to_numpy(), X_test_transformed, y_test.to_numpy()

        except Exception as e:
            logging.error(f"Error occurred in data transformation: {e}")
            raise CustomException(e, sys)

