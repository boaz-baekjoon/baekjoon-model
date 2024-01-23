import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os

client = QdrantClient("localhost", port=os.environ["QDRANT_PORT"])
client.create_collection(
    collection_name="items",
    vectors_config=VectorParams(size=40, distance=Distance.DOT),
)

item_embeds = np.load(os.environ["EMBED_FILE_LOC"])

operation_info = client.upsert(
    collection_name="items",
    wait=True,
    points=[
      PointStruct(id=idx, vector=list(item)) for idx, item in enumerate(item_embeds)
    ],
)

