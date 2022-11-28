import pymongo

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATA_FILE_PATH="/config/workspace/aps_fail_project/aps_failure_training_set1.csv"
DATABASE_NAME="aps"
COLLECTION_NAME="sensor"

if __name__ == "__main__":
   
    
    # # inser converted json record to mongo db
    data = client[DATABASE_NAME][COLLECTION_NAME].find()

    # Printing all records present in the collection
    for idx, record in enumerate(data):
          print(f"{idx}")
