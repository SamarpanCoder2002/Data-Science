from sensor.entity.config_entity import DataValidationConfig
from sensor.entity.artifact_entity import DataValidationArtifact
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
from typing import Option
import pandas as pd

import os
import sys


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_columns(self, df) -> Option[pd.DataFrame]:
        """_summary_

        This function will drop column which contains missing value more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None

        Raises:
            SensorException: _description_
        """

        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            
            # Selecting column name which contains null values
            drop_column_names = null_report[null_report.values > threshold].index
            
            # Dropping columns that have more than thresold null values
            df.drop(list(drop_column_names), axis=1, inplace=True)
            
            # return None when no columns left
            if len(df.columns) == 0:
                return None
            return df
            
            
                    
        except Exception as e:
            raise SensorException(e, sys)
        
    def is_required_columns_exists(self,base_df, present_df) -> Option[bool]:
        try:
            ...
        except Exception as e:
            raise SensorException(e, sys)
            
    def initiate_data_validation(self) -> DataValidationArtifact:
        ...
