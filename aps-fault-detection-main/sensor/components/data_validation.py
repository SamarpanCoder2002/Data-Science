from sensor.entity.config_entity import DataValidationConfig
from sensor.entity.artifact_entity import DataValidationArtifact
from sensor.exception import SensorException
from sensor.logger import logging
from sensor import utils
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
            self.validation_error = dict()
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
            drop_column_names = null_report[null_report.values >
                                            threshold].index

            self.validation_error['dropped_columns'] = drop_column_names

            # Dropping columns that have more than thresold null values
            df.drop(list(drop_column_names), axis=1, inplace=True)

            # return None when no columns left
            if len(df.columns) == 0:
                return None
            return df

        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, curr_df: pd.DataFrame) -> Option[bool]:
        try:
            base_columns = base_df.columns
            curr_columns = curr_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column in curr_columns:
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error['Missing columns'] = missing_columns
                return False

            return True

        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, curr_df: pd.DataFrame):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            curr_columns = curr_df.columns

            for base_column in base_columns:
                base_data, curr_data = base_df[base_column], curr_df[base_column]

                # Null hypothesis is that both column data drawn from same distribution

                same_distribution = ks_2samp(base_data, curr_data)

                if same_distribution.pvalue > 0.05:
                    # Accepting null hypothesis
                    drift_report[base_column] = {
                        'pvalues': same_distribution.pvalue,
                        'same_distribution': True
                    }
                else:
                    drift_report[base_column] = {
                        'pvalues': same_distribution.pvalue,
                        'same_distribution': False
                    }
                    # diff distribution

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            
            #
        except Exception as e:
            raise SensorException(e, sys)
