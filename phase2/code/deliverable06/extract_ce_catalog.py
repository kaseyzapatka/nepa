"""D6 — render the existing-CE catalog to Claude-readable Markdown.

Reads the canonical CE source (`ce_source.load_ce_catalog()` → `ce.json`, the CE
Explorer export) and writes a clean, agency-grouped Markdown catalog of existing
federal categorical exclusions, for cross-referencing D6 candidate categories
(so the deliverable surfaces only net-new / expand / adopt opportunities).

This supersedes the earlier xlsx-based extraction: ce.json is already one clean
record per CE, so no heuristic parsing is needed and no `openpyxl` dependency.

Source: CE Explorer (https://ce.permitting.innovation.gov/data/exclusions.json);
each CE carries its canonical eCFR `canonical_source_url`.

Output: phase2/notes/deliverable06/_ce_catalog_extracted.md
(underscore-prefixed so the Quarto site render skips it — internal reference, not a site page)
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

from pathlib import Path

import ce_source
from common import normalize_space

NOTES = Path(__file__).resolve().parents[2] / "notes" / "deliverable06"
MD_OUT = NOTES / "_ce_catalog_extracted.md"


def _has(value: object) -> bool:
    s = normalize_space(value)
    return bool(s) and s.lower() not in ("nan", "none", "not catalogued")


def main() -> None:
    df = ce_source.load_ce_catalog()
    version = ce_source.catalog_version()
    df = df.sort_values(["agency_unit", "structured_id"], na_position="last")

    lines = [
        "# Existing Federal Categorical Exclusions — catalog (for D6 cross-reference)",
        "",
        "**Source:** CE Explorer export (`ce.json`) — "
        f"{ce_source.CE_EXPLORER_URL}",
        f"**Catalog version:** {version.get('version', '?')} (dated {version.get('date', '?')})",
        f"**Existing CEs:** {len(df):,} across {df['agency_unit'].nunique()} agency units",
        "",
        "> CE Explorer is a discovery index; each entry links to its canonical eCFR "
        "source. Use this list to classify D6 candidates as **new** (no match here), "
        "**expand** (matches but our FONSIs exceed its bounds), or **adopt** (exists at "
        "another agency). The official CEQ government-wide list is a secondary "
        "authoritative cross-check.",
        "",
        "---",
        "",
    ]
    for unit, grp in df.groupby("agency_unit", sort=True):
        long = normalize_space(grp["agency_name"].iloc[0])
        header = f"## {unit}" + (f" — {long}" if long and long != unit else "")
        lines.append(f"{header}  ·  {len(grp)} CEs")
        lines.append("")
        for r in grp.itertuples(index=False):
            sid = normalize_space(r.structured_id) or "—"
            desc = normalize_space(r.ce_description)
            lines.append(f"- **[{sid}]** {desc}")
            if _has(r.extraordinary_circumstances):
                lines.append(f"  - *Extraordinary circumstances:* {normalize_space(r.extraordinary_circumstances)}")
            if _has(r.canonical_source_url):
                lines.append(f"  - *Source:* {normalize_space(r.canonical_source_url)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ce_catalog] {len(df):,} CEs across {df['agency_unit'].nunique()} units "
          f"(CE Explorer v{version.get('version','?')} {version.get('date','?')})")
    print(f"[ce_catalog] wrote {MD_OUT}")


if __name__ == "__main__":
    main()
