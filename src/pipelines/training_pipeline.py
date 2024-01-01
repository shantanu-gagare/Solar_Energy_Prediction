import os
print(os.getcwd())
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
sys.path.insert(0, os.path.abspath('E:/ML_Projects/Solar_energy_prediction_end_to_end/src'))

from components.data_ingest import DataIngestion

if __name__=="__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
