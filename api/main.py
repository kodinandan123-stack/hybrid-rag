"""FastAPI application exposing a /query endpoint for the hybrid RAG pipeline."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation.generator import Generator
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseRetriever

app = FastAPI(title="Hybrid RAG API")

_dense = DenseRetriever()
_sparse: SparseRetriever | None = None
_hybrid: HybridRetriever | None = None
_generator = Generator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


def _get_hybrid_retriever() -> HybridRetriever:
    if _hybrid is None:
        raise HTTPException(
            status_code=400,
            detail="Corpus not indexed yet; call POST /index first",
        )
    return _hybrid


@app.post("/index")
def index_chunks(chunks: list[dict[str, Any]]) -> dict[str, int]:
    """Index a batch of chunk dicts into both the dense and sparse retrievers."""
    if not chunks:
        raise HTTPException(status_code=400, detail="chunks must not be empty")
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
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}
