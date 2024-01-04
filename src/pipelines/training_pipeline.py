import os
print(os.getcwd())
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

sys.path.insert(0, os.path.abspath('E:/ML_Projects/Solar_energy_prediction_end_to_end/src'))

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation




if __name__=="__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transformation= DataTransformation()

    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_path=train_data_path, test_path= test_data_path)
