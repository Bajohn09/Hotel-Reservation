from src.logger import get_logger
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *
import sys

logger = get_logger(__name__)


if __name__ == "__main__":
    # data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    # data_ingestion.run()

    # processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    # processor.process()

    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()