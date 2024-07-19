import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from pydantic import BaseModel
from src.utils import load_object
from src.constant import MODEL_FILEPATH, PREPROCESSOR_FILEPATH

class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

class Predict:
    def __init__(self):
        # Load model and preprocessor during initialization
        try:
            self.preprocessor = load_object(PREPROCESSOR_FILEPATH)
            self.model = load_object(MODEL_FILEPATH)
        except Exception as e:
            logging.error(f"Error occurred during model or preprocessor initialization: {e}")
            raise CustomException(e, sys)

    def get_prediction(self, data: InputData):
        try:
            # Convert InputData to a DataFrame
            data_dict = data.dict()
            data_df = pd.DataFrame([data_dict])
            
            # Transform and predict
            transformed_data = self.preprocessor.transform(data_df)
            result = self.model.predict(transformed_data)
            return round(result[0], 2)

        except Exception as e:
            logging.error(f"Error occurred in prediction: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Sample data
    sample_data = InputData(
        gender='female', 
        race_ethnicity='group A',
        parental_level_of_education="bachelor's degree",
        lunch='standard',
        test_preparation_course='none', 
        reading_score=12,
        writing_score=2
    )
    
    # Initialize Predict and get prediction
    predict = Predict()
    try:
        result = predict.get_prediction(sample_data)
        print(result)
    except CustomException as e:
        print(f"Prediction failed: {e}")
