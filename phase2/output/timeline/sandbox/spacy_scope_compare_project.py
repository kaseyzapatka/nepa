import re
import pandas as pd
import spacy

PID = "c87a153c-f0c6-bd71-17e1-7e01ea9816a5"

DATE_RE = re.compile(
    r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\.\d{2}\.\d{2}\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}\b)",
    re.IGNORECASE,
)


def unwrap(value):
    return value.get("value") if isinstance(value, dict) else value


def first_root_verb(nlp, text: str) -> str:
    doc = nlp(text)
    for tok in doc:
        if tok.dep_ in ("ROOT", "relcl") and tok.pos_ == "VERB":
            return tok.text
    return "<no_verb>"


def main():
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "ner"])

    print("=== REGEX CACHE CONTEXTS ===")
    regex = pd.read_parquet("data/analysis/regex_candidates_ce.parquet")
    sub = regex[regex["project_id"] == PID].sort_values(["position", "date"]).reset_index(drop=True)
    for i, row in sub.iterrows():
        ctx = str(row["context"]).replace("\n", " ")
        print(f"{i + 1:02d}. {row['date']} | {first_root_verb(nlp, ctx)} | {ctx[:150]}")

    print()
    print("=== FULL MAIN DOCUMENT: DATE-BEARING SENTENCES ===")
    docs = pd.read_parquet("data/processed/ce/documents.parquet")
    docs["project_id"] = docs["project_id"].map(unwrap)
    main_doc_id = docs[(docs["project_id"] == PID) & (docs["main_document"] == "YES")].iloc[0]["document_id"]

    pages = pd.read_parquet("data/processed/ce/pages.parquet")
    page_df = pages[pages["document_id"] == main_doc_id].copy()
    page_df["page_sort"] = page_df["page_number"].astype(str).str.extract(r"(\d+)").fillna("999999").astype(int)
    page_df = page_df.sort_values(["page_sort", "page_number"])

    count = 0
    for _, page in page_df.iterrows():
        page_num = page["page_number"]
        doc = nlp(str(page["page_text"]))
        for sent in doc.sents:
            sent_text = " ".join(sent.text.split())
            if DATE_RE.search(sent_text):
                count += 1
                print(f"p{page_num} #{count:02d}. {first_root_verb(nlp, sent_text)} | {sent_text[:180]}")


if __name__ == "__main__":
    main()
