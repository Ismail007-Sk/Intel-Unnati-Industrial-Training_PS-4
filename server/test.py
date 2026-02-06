from pymongo import MongoClient

uri = "MongoDB_URI"
client = MongoClient(uri, serverSelectionTimeoutMS=5000)
print(client.list_database_names())
