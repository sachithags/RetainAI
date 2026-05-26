import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import numpy as np
import pandas as pd
import pymongo
from retainAI.exception.exception import RetainAIException
from retainAI.logging.logger import logging

class RetainAI_DataExtracter():
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            logging.info("MongoDB client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing MongoDB client: {e}")
            raise RetainAIException(e, sys)
        
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = json.loads((data.to_json(orient="records")))
            return records
        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {e}")
            raise RetainAIException(e, sys)
        
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.records = records
            self.database= database
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
            logging.info("Data inserted successfully into MongoDB.")
        except Exception as e:
            logging.error(f"Error inserting data into MongoDB: {e}")
            raise RetainAIException(e, sys)
        
if __name__ == "__main__":
    FILE_PATH = "RetainAI_data/HR-Employee-Attrition.csv"
    DATABASE = "RetainAI_DB"
    Collection = "RetainAIData"
    networkobj = RetainAI_DataExtracter()
    records = networkobj.csv_to_json_converter(file_path = FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records,DATABASE, Collection)
    print(f"{no_of_records} records inserted successfully into MongoDB.")