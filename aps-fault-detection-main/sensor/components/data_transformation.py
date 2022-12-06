from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utils
from sensor.entity import artifact_entity, config_entity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sensor import config

import pandas as pd
import numpy as np

import os
import sys


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig, data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config

            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object() -> Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='contant', fill_value=0)
            robust_scaler = RobustScaler()

            pipeline = Pipeline(
                steps=[
                    ('Imputer', simple_imputer),
                    ('RobustScaler', robust_scaler)
                ]
            )

            return pipeline

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info('Reading training and testing file')
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info('Selecting input feature from train and test df')
            input_feature_train_df = train_df.drop(
                config.TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(config.TARGET_COLUMN, axis=1)

            logging.info('Seleting target feature for train and test df')
            target_feature_train_df = train_df[config.TARGET_COLUMN]
            target_feature_test_df = test_df[config.TARGET_COLUMN]

            # In some columns there are some non-numeric values.
            # We have to convert them to the numeric values for further work
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # transformation on target columns
            target_feature_train_arr = label_encoder.transform(
                target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(
                target_feature_test_df)

            # Now we have to transform our input features
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # Transformation on input columns
            input_feature_train_arr = transformation_pipeline.transform(
                input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(
                input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="minority")
            logging.info(
                f'Before resampling in training set Input: {input_feature_train_arr.shape}     Target: {target_feature_train_arr.shape}')

            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(
                input_feature_train_arr, target_feature_train_arr)

            logging.info(
                f'After resampling in training set Input: {input_feature_train_arr.shape}     Target: {target_feature_train_arr.shape}')

            logging.info(
                f'Before resampling in test set Input: {input_feature_test_arr.shape}     Target: {target_feature_test_arr.shape}')

            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(
                input_feature_test_arr, target_feature_test_arr)

            logging.info(
                f'After resampling in test set Input: {input_feature_test_arr.shape}     Target: {target_feature_test_arr.shape}')

            # Now we have to save out target encoder
            logging.info('Merge two arrays')
            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save numpy array
            logging.info('Save Combined train and test array')
            utils.save_numpy_arr_data(
                file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_arr_data(
                file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            logging.info('Now we have to store our pipeline')
            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            logging.info(f'Storing label encoder')
            utils.save_object(
                file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            logging.info(f'Getting data transformation artifact')
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path
            )
            
            logging.info(f'Data transformation object {data_transformation_artifact}')
            
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
