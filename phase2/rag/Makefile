.PHONY: rag-smoke-store rag-smoke-index rag-smoke-build rag-store rag-chunks rag-index rag-index-skip-fts rag-build rag-eval rag-app

rag-smoke-store:
	conda run -n nepa python phase2/rag/scripts/01_build_text_store.py --process-types CE --sample-documents-per-process 2

rag-smoke-index:
	conda run -n nepa python phase2/rag/scripts/03_build_indexes.py --skip-fts

rag-smoke-build: rag-smoke-store rag-chunks rag-smoke-index

rag-store:
	conda run -n nepa python phase2/rag/scripts/01_build_text_store.py

rag-chunks:
	conda run -n nepa python phase2/rag/scripts/02_build_chunks.py

rag-index:
	conda run -n nepa python phase2/rag/scripts/03_build_indexes.py

rag-index-skip-fts:
	conda run -n nepa python phase2/rag/scripts/03_build_indexes.py --skip-fts

rag-build: rag-store rag-chunks rag-index

rag-eval:
	conda run -n nepa python phase2/rag/scripts/04_eval_rag.py

rag-app:
	conda run -n nepa streamlit run phase2/rag/app/app.py
