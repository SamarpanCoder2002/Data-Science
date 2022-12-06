from sensor.entity.config_entity import DataValidationConfig, DataIngestionConfig
from sensor.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from sensor.exception import SensorException
from sensor.logger import logging
from sensor import utils
from scipy.stats import ks_2samp
from typing import Optional
import pandas as pd
import numpy as np
from sensor import config

import os
import sys


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_ingestion_config = data_ingestion_config
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_columns(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
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
            logging.info(
                f'Selecting column names contains null values above to {threshold}')
            drop_column_names = null_report[null_report.values >
                                            threshold].index

            self.validation_error[report_key_name] = list(drop_column_names)

            # Dropping columns that have more than thresold null values
            logging.info(
                f'Dropping columns that have more than thresold null values: {drop_column_names}')
            df.drop(list(drop_column_names), axis=1, inplace=True)

            # return None when no columns left
            logging.info('return None when no columns left')
            if len(df.columns) == 0:
                return None
            return df

        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, curr_df: pd.DataFrame, report_key_name: str) -> Optional[bool]:
        try:
            base_columns = base_df.columns
            curr_columns = curr_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column not in curr_columns:
                    logging.info(f"Column: [{base_column}]  is not available")
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False

            return True

        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, curr_df: pd.DataFrame, report_key_name: str):
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
                        'pvalues': float(same_distribution.pvalue),
                        'same_distribution': True
                    }
                else:
                    drift_report[base_column] = {
                        'pvalues': float(same_distribution.pvalue),
                        'same_distribution': False
                    }
                    # diff distribution

            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info(f"Reading base df: {self.data_validation_config.base_file_path}")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"Replace na value with np.NAN")
            base_df.replace({"na": np.NAN}, inplace=True)

            logging.info(f"Dropping missing values")
            base_df = self.drop_missing_columns(
                df=base_df, report_key_name='missing_values_within_base_dataset')

            logging.info(f"Reading train and test df")
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Drop null values from train and test df")
            train_df = self.drop_missing_columns(
                df=train_df, report_key_name='missing_values_within_train_dataset')
            test_df = self.drop_missing_columns(
                df=test_df, report_key_name='missing_values_within_test_dataset')

            logging.info(
                f"Is all required columns present in train and test df")
            train_df_columns_status = self.is_required_columns_exists(
                base_df=base_df, curr_df=train_df, report_key_name='missing_columns_within_train_dataset')
            test_df_columns_status = self.is_required_columns_exists(
                base_df=base_df, curr_df=test_df, report_key_name='missing_columns_within_test_dataset')
            
            exclude_columns = [config.TARGET_COLUMN]
            base_df=utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df=utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df=utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            if train_df_columns_status:
                logging.info(
                    f'As all columns are available in train data drift')
                self.data_drift(base_df=base_df, curr_df=train_df,
                                report_key_name='data_drift_within_train_dataset')

            if test_df_columns_status:
                logging.info(
                    f'As all columns are available in test data drift')
                self.data_drift(base_df=base_df, curr_df=test_df,
                                report_key_name='data_drift_within_test_dataset')

            # write the report
            logging.info(f'Report writing')
            utils.write_yaml_file(
                file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            logging.info(f'Getting data validation artifact')
            data_validation_artifact = DataValidationArtifact(
                self.data_validation_config.report_file_path)

            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)
