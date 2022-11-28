from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataTransformationArtifact:
        try:

            logging.info("Exporting collection data as pandas dataframe")
            # Exporting collection data as pandas dataframe
            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)

            # save data in feature store
            logging.info("# save data in feature store")
            df.replace(to_replace='na', value=np.NAN, inplace=True)

            # feature store
            logging.info("feature store")
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            # Save df to feature store folder
            logging.info("Save df to feature store folder")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,
                      index=False, header=True)

            logging.info('split dataset into train and test set')
            # split dataset into train and test set
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size)

            # create dataset directory folder if not available
            logging.info("Create dataset dir folder if not available")
            dataset_dir = os.path.dirname(
                self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            # save df to feature store folder
            logging.info("save df to feature store folder")
            train_df.to_csv(
                path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact
            logging.info("Prepare artifact")
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                                                                            train_file_path=self.data_ingestion_config.train_file_path, test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
