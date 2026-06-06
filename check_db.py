import os
from dotenv import load_dotenv
import pymongo

load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
print('MongoDB URL found:', bool(MONGO_DB_URL))

client = pymongo.MongoClient(MONGO_DB_URL)

# Replace these with your actual database and collection names
# (check your constants file)
DATABASE_NAME = "RetainAI"
COLLECTION_NAME = "HR_Attrition"

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
count = collection.count_documents({})
print(f'Documents in collection: {count}')

if count > 0:
    sample = collection.find_one()
    print('Sample keys:', list(sample.keys())[:10])