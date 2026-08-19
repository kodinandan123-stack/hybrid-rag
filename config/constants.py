"""Shared constants for the hybrid-RAG pipeline."""

# Chunking
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Retrieval
TOP_K_DENSE: int = 10
TOP_K_SPARSE: int = 10
RRF_K: int = 60
RERANK_TOP_N: int = 5

# Models
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Evaluation
EVAL_TESTSET_PATH: str = "eval/testset.jsonl"
EVAL_RESULTS_DIR: str = "eval/results"
