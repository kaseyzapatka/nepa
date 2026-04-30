# Document Explorer Deployment (HF Spaces)

**Purpose:** Deploy the Streamlit NEPA document explorer to Hugging Face Spaces without storing the 7+ GB DuckDB file inside the Space repo.
**Strategy:** Build DuckDB locally → upload to an HF Dataset repo → deploy a Docker Space that downloads the DB at runtime.
**Prerequisites:** `huggingface_hub` CLI installed and authenticated (`huggingface-cli login`).

---

## Step 1 — Build the DuckDB locally (one-time per data refresh)

```bash
python code/rag/01_build_text_store.py
```

Output: `data/rag/nepa_reader.duckdb`

---

## Step 2 — Upload the DB to a Hugging Face Dataset repo

Set your values:

```bash
HF_USERNAME="YOUR_HF_USERNAME"
DB_REPO="nepa-document-explorer-db"
```

Create the dataset repo (safe to rerun):

```bash
hf repo create "${HF_USERNAME}/${DB_REPO}" --repo-type dataset || true
```

Upload the DB file:

```bash
hf upload "${HF_USERNAME}/${DB_REPO}" data/rag/nepa_reader.duckdb nepa_reader.duckdb --repo-type dataset
```

---

## Step 3 — Deploy app to a Hugging Face Docker Space

Set Space name:

```bash
SPACE_NAME="nepa-document-explorer"
```

Create Space (if CLI supports `--space_sdk`):

```bash
hf repo create "${HF_USERNAME}/${SPACE_NAME}" --repo-type space --space_sdk docker || true
```

> If your CLI does not accept `--space_sdk`, create the Space in the HF web UI as **Docker**, then continue.

Prepare a clean deploy folder:

```bash
DEPLOY_DIR="$(mktemp -d)"
cp app/app.py "${DEPLOY_DIR}/app.py"
cp app/requirements.txt "${DEPLOY_DIR}/requirements.txt"
```

Create `Dockerfile`:

```bash
cat > "${DEPLOY_DIR}/Dockerfile" <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV NEPA_DB_HF_REPO=YOUR_HF_USERNAME/nepa-document-explorer-db
ENV NEPA_DB_HF_FILENAME=nepa_reader.duckdb
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=7860"]
EOF
```

Upload app to Space:

```bash
hf upload "${HF_USERNAME}/${SPACE_NAME}" "${DEPLOY_DIR}" --repo-type space --commit-message "Deploy NEPA document explorer"
```

---

## Step 4 — Verify

Space URL: `https://huggingface.co/spaces/YOUR_HF_USERNAME/nepa-document-explorer`

The Quarto navbar link is configured in `_quarto.yml`:
`Document Explorer -> https://huggingface.co/spaces/<username>/nepa-document-explorer`

---

## Routine updates

- **App-only update:** re-upload `app.py`, `requirements.txt`, `Dockerfile` to the Space repo.
- **Data refresh:** rebuild `nepa_reader.duckdb` (step 1), upload to the dataset repo (step 2), then restart/rebuild the Space.

---

## Notes

- HF Space repos have strict storage limits on the free tier (~1 GB). Do **not** commit `.duckdb` into the Space repo.
- Keeping the DB in a dataset repo avoids Git LFS in the Space deployment flow.
