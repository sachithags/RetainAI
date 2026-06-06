from retainAI.exception.exception import RetainAIException
from retainAI.logging.logger import logging
import os
import sys


from retainAI.entity.config_entity import DataIngestionConfig
from retainAI.entity.artifact_entity import DataIngestionArtifact

import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd


from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info(f"Data Ingestion config: {self.data_ingestion_config}")
        except Exception as e:
            logging.error(f"Error initializing DataIngestion: {e}")
            raise RetainAIException(e, sys)
        

    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if"_id" in df.columns.to_list():
                df.drop("_id", axis=1, inplace=True)

            df.replace(to_replace="na", value=np.nan, inplace=True)
            return df
        except Exception as e:
            raise RetainAIException(e, sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
        except Exception as e:
            raise RetainAIException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            # Create directories for both train and test files
            dir_path_train = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path_train, exist_ok=True)
            dir_path_test = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(dir_path_test, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
        except Exception as e:
            raise RetainAIException(e, sys)

    def initiate_data_ingestion(self):
        try:
            # 1. Get DataFrame from MongoDB
            dataframe = self.export_collection_as_dataframe()
            
            # 2. Save raw data to feature store (don't overwrite the DataFrame)
            self.export_data_into_feature_store(dataframe)
            
            # 3. Split into train/test (dataframe is still the real DataFrame)
            self.split_data_as_train_test(dataframe)
            
            # 4. Build and return the artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise RetainAIException(e, sys)


