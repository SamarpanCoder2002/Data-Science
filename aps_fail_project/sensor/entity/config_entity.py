import os

FILE_NAME = "sensor.csv"


class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(
            os.getcwd(), 'artifact', f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.database = "aps"
        self.collection_name = "sensor"
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir, "data_ingestion")
        self.feature_store_dir = os.path.join(
            self.data_ingestion_dir, 'feature_store', FILE_NAME)


class DataValidationConfig:
    ...


class DataTransformationConfig:
    ...


class ModelTrainerConfig:
    ...


class ModelEvaluationConfig:
    ...


class ModelPusherConfig:
    ...
