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
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

import os
import sys


class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig, data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_tranformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
    def fine_tune(self):
        try:
            ...
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x, y):
        xgb_clf = XGBClassifier()
        xgb_clf.fit(x, y)
        return xgb_clf

    def initiate_model_trainer(self,) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f'Loading train and test array')
            train_arr = utils.load_numpy_arr_data(
                file_path=self.data_tranformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_arr_data(
                file_path=self.data_tranformation_artifact.transformed_test_path)

            logging.info(
                f'Splitting input and target feature from train and test dataset')
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f'Train the model')
            model = self.train_model(x_train, y_train)

            logging.info(f'Calculating f1 train score')
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f'Calculating f1 test score')
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(
                f'f1 train score: {f1_train_score} and f1 test score: {f1_test_score}')

            # Check for overfitting or underfitting
            logging.info(f'Checking if our model is underfitted or not')
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(
                    f'Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}:    model actual score: {f1_test_score}')

            logging.info('Checking if the model is overfitted  or not')
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_thresold:
                raise Exception(
                    f'Train and test score diff: {diff} is more than overfitting thresold ${self.model_trainer_config.overfitting_thresold}')

            logging.info(f'Saving the training model')
            utils.save_object(
                file_path=self.model_trainer_config.model_path, obj=model)

            logging.info('Preparaing artifact')
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                f1_train_score=f1_train_score,
                f1_test_score=f1_test_score
            )
            logging.info(f'Model Trainer Artifact: {model_trainer_artifact}')
            
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)
