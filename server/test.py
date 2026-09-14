from pymongo import MongoClient
import certifi
import os
from dotenv import load_dotenv


load_dotenv()


MONGO_URI=os.getenv("MONGO_URI")

print("MONGO_URI exists:", bool(MONGO_URI))


client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where()
)

print(client.admin.command("ping"))