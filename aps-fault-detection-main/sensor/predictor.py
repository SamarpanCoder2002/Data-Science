from sensor.exception import SensorException
import os
import sys
from sensor.entity import config_entity


class ModelResolver:
    def __init__(self, model_registry: str = 'saved_models', tarnsformer_dir_name='transformer', target_encoder_dir_name='target_encoder', model_dir_name='model'):
        self.model_registry = model_registry
        os.makedirs(model_registry, exist_ok=True)
        self.transformer_dir_name = tarnsformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    def get_latest_dir_path(self,):
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            
            dir_names = list(map(int, dir_names))
            latest_folder_name = max(dir_names)

            return os.path.join(self.model_registry, f'{latest_folder_name}')

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f'Model is not available')

            return os.path.join(latest_dir, self.model_dir_name, config_entity.MODEL_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f'Transformer is not available')

            return os.path.join(latest_dir, self.transformer_dir_name, config_entity.TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f'Transformer is not available')

            return os.path.join(latest_dir, self.target_encoder_dir_name, config_entity.TARGET_ENCODER_OBJECT_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_dir_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, f'{0}')

            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, f'{latest_dir_num+1}')

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception('Save model path not found')

            return os.path.join(latest_dir, self.model_dir_name, config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception('Save transformer path not found')

            return os.path.join(latest_dir, self.transformer_dir_name, config_entity.TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            if latest_dir is None:
                raise Exception('Target encoder path not found')

            return os.path.join(latest_dir, self.target_encoder_dir_name, config_entity.TARGET_ENCODER_OBJECT_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)


class Predictor:
    def __init__(self, model_resolver: ModelResolver):
        self.model_resolver = model_resolver
