import os
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
DATABASE_NAME = "RetainAI"          # Make sure this matches your constants
COLLECTION_NAME = "HR_Attrition"    # Make sure this matches your constants

# Load the CSV – rename column to match your target column name
df = pd.read_csv("RetainAI_data/HR_Employee_Attrition.csv")
df.rename(columns={"Attrition": "Churn"}, inplace=True)

client = pymongo.MongoClient(MONGO_DB_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

records = df.to_dict(orient='records')
collection.insert_many(records)
print(f"Successfully inserted {len(records)} documents into {DATABASE_NAME}.{COLLECTION_NAME}")