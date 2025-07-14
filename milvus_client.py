from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

def connect_milvus():
    connections.connect(alias="default", host="localhost", port="19530")

    if utility.has_collection("faces"):
        return Collection("faces")

    fields = [
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]

    schema = CollectionSchema(fields, description="Face embeddings collection")
    collection = Collection(name="faces", schema=schema)
    return collection
