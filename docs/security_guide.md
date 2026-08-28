# Security Guide

This document covers authentication, authorisation, and data-privacy practices for the Hybrid RAG service.

## API Authentication

All API endpoints require a bearer token passed in the `Authorization` header:

```
Authorization: Bearer <token>
```

Tokens are issued per service account and rotated every 90 days. Store them in a secrets manager (e.g. HashiCorp Vault or AWS Secrets Manager) — never hard-code them in source files.

## Rate Limiting

The API enforces per-token rate limits to prevent abuse:

| Tier     | Requests / minute |
|----------|-------------------|
| Free     | 20                |
| Standard | 200               |
| Premium  | 2 000             |

Exceeding the limit returns HTTP 429. Clients should implement exponential back-off with jitter.

## Data Privacy

- **PII scrubbing**: Run uploaded documents through the PII scrubber (`ingestion/scrubber.py`) before indexing.
- **Retention**: Raw documents are deleted from the ingest queue within 24 hours of successful indexing.
- **Encryption at rest**: Vector embeddings are stored on AES-256-encrypted volumes.
- **Encryption in transit**: All traffic is served over TLS 1.3.

## Secrets Management

Never commit secrets to version control. Use environment variables or a secrets manager:

```bash
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
```

Add a pre-commit hook (`scripts/check_secrets.sh`) to scan for accidental secret leaks.

## Dependency Scanning

Run `pip-audit` in CI to flag known CVEs in Python dependencies:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

## Reporting Vulnerabilities

Disclose security issues privately to the maintainers via GitHub Security Advisories before making any public disclosure.
