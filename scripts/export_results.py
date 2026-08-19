"""export_results.py - batch-export hybrid-RAG query results to JSONL."""
from __future__ import annotations
import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path

def _post(api_url, query, top_k):
    payload = json.dumps({"query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(api_url + "/query", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.add_argument("--api-url", default="http://localhost:8000", dest="api_url")
    args = p.parse_args(argv)
    queries = [q.strip() for q in Path(args.queries).read_text().splitlines() if q.strip()]
    errors = 0
    with Path(args.output).open("w", encoding="utf-8") as fh:
        for query in queries:
            t0 = time.perf_counter()
            try:
                result = _post(args.api_url, query, args.top_k)
                rec = {"query": query, "answer": result.get("answer",""),
                       "chunks": result.get("chunks",[]),
                       "latency_ms": round((time.perf_counter()-t0)*1000,1)}
            except Exception as exc:
                rec = {"query": query, "answer": None, "chunks": [],
                       "latency_ms": round((time.perf_counter()-t0)*1000,1), "error": str(exc)}
                errors += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0 if errors == 0 else 1


def export_csv(jsonl_path: str, csv_path: str) -> None:
        """Convert an existing JSONL results file to CSV for spreadsheet analysis."""
        import csv
        rows = []
        with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                                    rec = json.loads(line)
                                    rows.append({
                                                        "query": rec.get("query", ""),
                                                        "answer": rec.get("answer", ""),
                                                        "latency_ms": rec.get("latency_ms", ""),
                                                        "num_chunks": len(rec.get("chunks", [])),
                                                        "error": rec.get("error", ""),
                                    })
                            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                                        writer = csv.DictWriter(f, fieldnames=["query", "answer", "latency_ms", "num_chunks", "error"])
                                        writer.writeheader()
                                        writer.writerows(rows)


if __name__ == "__main__":
        raise SystemExit(main())

if __name__ == "__main__":
    sys.exit(main())
