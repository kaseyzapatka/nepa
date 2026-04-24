# Hugging Face Space Notes

Deployment is on hold until the local RAG build is validated.

When ready, use a separate Hugging Face Dataset repo for Phase 2 RAG artifacts:

```text
YOUR_HF_USERNAME/nepa-phase2-rag-db
```

The Space should download or mount:

- `manifest.json`
- `rag_catalog.duckdb`
- page parquet shards
- chunk parquet shards
- per-shard DuckDB FTS files

The model does not need to run inside the Space for the first hosted pilot. The Streamlit app can call a separate model endpoint through `LLM_PROVIDER`, `LLM_MODEL`, `LLM_QUALITY_MODEL`, `LLM_FALLBACK_MODEL`, and `OLLAMA_HOST`.
