# API Reference

This document describes all HTTP endpoints exposed by the Hybrid RAG service.

Base URL: `http://localhost:8000`

---

## Health

### GET /health

Liveness probe. Returns `200 OK` if the process is running.

**Response**
```json
{
  "status": "ok",
    "timestamp": "2026-08-23T10:00:00+00:00"
    }
    ```

    ### GET /health/ready

    Readiness probe. Returns system resource metrics alongside status.

    **Response**
    ```json
    {
      "status": "ready",
        "timestamp": "2026-08-23T10:00:00+00:00",
          "system": {
              "python": "3.11.4",
                  "os": "Linux",
                      "cpu_count": 8,
                          "memory_used_pct": 42.1,
                              "disk_used_pct": 31.7
                                }
                                }
                                ```

                                ---

                                ## Retrieval

                                ### POST /retrieve

                                Run a hybrid retrieval query using Reciprocal Rank Fusion (RRF) over dense and sparse indexes.

                                **Request body**
                                ```json
                                {
                                  "query": "string",
                                    "top_k": 10,
                                      "alpha": 0.5
                                      }
                                      ```

                                      | Field | Type | Required | Description |
                                      |-------|------|----------|-------------|
                                      | `query` | string | yes | The natural-language query |
                                      | `top_k` | int | no (default 10) | Number of results to return |
                                      | `alpha` | float | no (default 0.5) | Weight for dense vs sparse scores (0 = sparse only, 1 = dense only) |

                                      **Response**
                                      ```json
                                      {
                                        "results": [
                                            {
                                                  "id": "doc42",
                                                        "text": "Relevant passage text...",
                                                              "rrf_score": 0.0312,
                                                                    "dense_rank": 1,
                                                                          "sparse_rank": 3
                                                                              }
                                                                                ],
                                                                                  "query": "original query",
                                                                                    "latency_ms": 87
                                                                                    }
                                                                                    ```

                                                                                    ---

                                                                                    ## Ingestion

                                                                                    ### POST /ingest

                                                                                    Ingest one or more documents into the dense and sparse indexes.

                                                                                    **Request body**
                                                                                    ```json
                                                                                    {
                                                                                      "documents": [
                                                                                          {
                                                                                                "id": "doc1",
                                                                                                      "text": "Document content here.",
                                                                                                            "metadata": {}
                                                                                                                }
                                                                                                                  ]
                                                                                                                  }
                                                                                                                  ```
                                                                                                                  
                                                                                                                  **Response**
                                                                                                                  ```json
                                                                                                                  {
                                                                                                                    "ingested": 1,
                                                                                                                      "failed": 0,
                                                                                                                        "duration_ms": 210
                                                                                                                        }
                                                                                                                        ```
                                                                                                                        
                                                                                                                        ---
                                                                                                                        
                                                                                                                        ## Error responses
                                                                                                                        
                                                                                                                        All endpoints return standard error bodies on failure:
                                                                                                                        
                                                                                                                        ```json
                                                                                                                        {
                                                                                                                          "detail": "Description of the error"
                                                                                                                          }
                                                                                                                          ```
                                                                                                                          
                                                                                                                          | Status | Meaning |
                                                                                                                          |--------|---------|
                                                                                                                          | 400 | Bad request — invalid parameters |
                                                                                                                          | 422 | Unprocessable entity — schema validation failed |
                                                                                                                          | 500 | Internal server error |
                                                                                                                          
