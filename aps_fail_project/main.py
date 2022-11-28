from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
from sensor.utils.dbinfo import DBClass
import os,sys
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity import config_entity


if __name__ == '__main__':
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        print(data_ingestion_config.to_dict())
    except Exception as e:
        print(e)
