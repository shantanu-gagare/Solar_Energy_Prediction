from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

# pipelines 
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np 

from src.exception import CustomException
from src.logger import logging
sys.path.append(os.path.abspath('E:/ML_Projects/Solar_energy_prediction_end_to_end/src'))
from utils import save_object



# Data transformation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')



# Data Transformation class 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        
        try:
            logging.info("Data Transformation initiated")

            # Define which columns should be ordinal encoded and which columns should be scaled
            cat_cols = ['Location', 'Season']
            numerical_cols = ['Date', 'Time', 'Latitude', 'Longitude', 'Altitude', 'YRMODAHRMI',
                            'Month', 'Hour', 'Humidity', 'AmbientTemp', 'Wind.Speed', 'Visibility',
                            'Pressure', 'Cloud.Ceiling']
            
            # Define ranking for categorical coumns
            season_categories = ['Winter', 'Fall', 'Spring', 'Summer']
            location_categories = ['Grissom', 'Malmstrom', 'MNANG', 'Camp Murray', 'Peterson', 'USAFA',
                'Travis', 'March AFB', 'Offutt', 'Hill Weber', 'Kahului', 'JDMT']
            
            logging.info("Pipeline Initiated")

            # Numerical Pipeline 
            num_pipeline = Pipeline(
                            steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                            ]
                        )
            
            # Column pipeline
            cat_pipeline = Pipeline(
                            steps=[
                                ('ordincalencoder', OrdinalEncoder(categories=[location_categories, season_categories])),
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('scaler', StandardScaler())
                            ]
                        )
            
            # to combine both categorical pipeline and numerical pipeline we use the column transformer 
            preprocessor = ColumnTransformer(
                [
                    ('cat_pipeline', cat_pipeline, cat_cols),
                    ('num_pipeline', num_pipeline,numerical_cols)
                    
                ]
            )
            logging.info('Pipeline completed')

            return preprocessor
            

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)    


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_object = self.get_data_transformation_object()

            target_column_name = "PolyPwr"

            drop_columns = [target_column_name]
            
            # dividing independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df['PolyPwr']

            # apply transformation
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            logging.info("Applying preprocessing the on test and train dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object

            )
            logging.info("Preprocessor pickle is created and saved ")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error occured in Initiate data Transformation")
            raise CustomException(e, sys)