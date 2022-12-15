from sensor.pipeline import training_pipeline
from sensor.pipeline import batch_prediction

file_path = 'aps_failure_training_set1.csv'

if __name__ == "__main__":
    try:
        #training_pipeline.start_training_pipeline()
        output_file=batch_prediction.start_batch_prediction(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        print(e)
