import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN: str = "Attrition"
PIPELINE_NAME: str = "RetainAI"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "HR_Employee_Attrition.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

DATA_INGESTION_COLLECTION_NAME: str = "RetainAIData"
DATA_INGESTION_DATABASE_NAME: str = "RetainAI_DB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion_dir"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

