"""FastAPI application exposing a /query endpoint for the hybrid RAG pipeline."""

from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator

app = FastAPI(title="Hybrid RAG API")

_dense = DenseRetriever()
_sparse: Optional[SparseRetriever] = None
_hybrid: Optional[HybridRetriever] = None
_generator = Generator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


def _get_hybrid_retriever() -> HybridRetriever:
    if _hybrid is None:
        raise RuntimeError("Corpus not indexed yet; call POST /index first")
    return _hybrid


@app.post("/index")
def index_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Index a batch of chunk dicts into both the dense and sparse retrievers."""
    global _sparse, _hybrid
    _dense.index(chunks)
    _sparse = SparseRetriever(chunks)
    _hybrid = HybridRetriever(dense=_dense, sparse=_sparse)
    return {"indexed": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a query by retrieving context with hybrid search and generating a grounded answer."""
    retriever = _get_hybrid_retriever()
    hits = retriever.search(request.query, top_k=request.top_k)
    answer = _generator.generate(request.query, hits)
    return QueryResponse(answer=answer, sources=hits)


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}
