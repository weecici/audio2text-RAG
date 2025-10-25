from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
# client.delete_collection(collection_name="documents")
collections = client.get_collections().collections
print(f"Existing Qdrant collections: {collections}")
if "cranfield" in [col.name for col in collections]:
    print(client.get_collection("cranfield"))
