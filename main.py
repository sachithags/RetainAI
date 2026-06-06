from retainAI.components.data_ingestion import DataIngestion
from retainAI.components.data_validation import DataValidation
from retainAI.exception.exception import RetainAIException
from retainAI.logging.logger import logging
from retainAI.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig
import sys

if __name__ == "__main__":
    try:
        # 1. Create configs
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        
        # 2. Start data ingestion
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Data Ingestion started")
        
        # 3. Run and get artifact
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # 4. Log and print the artifact (now the variable exists)
        logging.info(f"Data Ingestion Completed")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact,
            data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation Completed")
        print(data_validation_artifact)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise RetainAIException(e, sys)