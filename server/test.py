from pymongo import MongoClient
import certifi

client = MongoClient(
    "mongodb+srv://Ismail:ISMAIL21JUNE2026@cluster0.0rioeu8.mongodb.net/?appName=Cluster0",
    tls=True,
    tlsCAFile=certifi.where()
)

print(client.admin.command("ping"))