"""Dense retrieval using sentence-transformers embeddings and Qdrant vector search."""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class DenseRetriever:
    """Embeds chunks with sentence-transformers and indexes them in Qdrant."""

    def __init__(self, collection_name: str = "hybrid_rag_chunks", model_name: str = "all-MiniLM-L6-v2", url: str = "http://localhost:6333"):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def _ensure_collection(self, vector_size: int) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self._ensure_collection(vector_size=embeddings.shape[1])
        points = [
            PointStruct(id=idx, vector=embedding.tolist(), payload=chunk)
            for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [{"score": r.score, **r.payload} for r in results]
