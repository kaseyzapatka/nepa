# Federal Funding Mechanism Details


## Current Schema And Cues

*Updated 2026-07-25 from the published output (`federal_funding_detail_summary.csv`, n = 9,210
Funding-primary projects; mechanism priority order in `FUNDING_MECHANISM_PRIORITY`,
`01_extract_nepa_trigger.py` lines ~1145–1155).*

| Label | Count | Logic | Examples of text cues | Summary |
|---|---:|---|---|---|
| EERE Grant (PMC-ND Form) | 3,350 | Matches the EERE PMC-ND standard-form grant boilerplate (regex `pmc_nd_form`). | PMC-ND form headers and standard EERE grant-determination language. | DOE EERE grants issued on the standardized PMC-ND CX form — the single largest mechanism; split out from Grant/Award because the form itself identifies the instrument. |
| Grant/Award | 2,717 | Matches explicit grant or award terms (no higher-priority mechanism). | `grant`, `grants`, `award`, `awards`, `DOE grant`, `federal grant`, `grant award`, `selected for funding`. | A project where the evidence identifies a grant or award, but not a more specific higher-priority mechanism. |
| Unknown Funding Type | 2,091 | Funding-triggered, but no recognized mechanism cue in the funding evidence windows. | Blank sidecar evidence; FOA header only; manual funding label without a mechanism. | Funding may be real, but the instrument type was not identified. Treat as unresolved, not as a meaningful mechanism. |
| Formula Grant | 577 | Matches formula grant language or EECBG. | `formula grant`, `formula-based grant`, `formula award`, `EECBG`, `State Energy Program Formula Grants`. | A grant allocated by formula, often through DOE block-grant or state-energy-program channels. |
| ARPA-E Award | 200 | Matches ARPA-E award language (regex `arpa_e`). | `ARPA-E`, `Advanced Research Projects Agency–Energy` award language. | ARPA-E research awards, split out from Grant/Award by the agency-specific instrument language. |
| Revolving Loan | 130 | Matches revolving loan language. | `revolving loan`, `Revolving Loan Fund`, `Energy Bank Revolving Loan Program`. | A loan mechanism where repayments replenish a fund for future lending. |
| Loan Guarantee | 60 | Matches loan guarantee language. | `loan guarantee`, `guaranteed loan`, `issue a loan guarantee`. | The federal role is backing or guaranteeing a loan rather than directly granting funds. |
| Cooperative Agreement | 58 | Matches cooperative agreement language. | `cooperative agreement`, `under the terms of the cooperative agreement`. | A specific federal assistance instrument with substantial agency involvement. Outranks cost-share if both are present. |
| Financial Assistance | 10 | Matches the formal phrase `financial assistance` with no more specific instrument. | `financial assistance`, `active financial assistance agreement`. | Formal federal financial-assistance language, not specific enough to name the instrument. |
| Cost Share | 9 | Matches cost-share language; primary only if no higher-priority mechanism found. | `cost share`, `cost-shared arrangement`, `federal cost share`. | A financing structure where federal and non-federal parties split costs; most cost-share projects are counted under a more specific mechanism. |
| Generic Funding | 8 | Matches broad funding language but no explicit instrument (lowest priority). | `DOE would provide funds`, `federal funding`, `would receive federal funds`. | Clearly federal money, but no instrument named and no other cue matched — now rare because PMC-ND/ARPA-E capture most former members of this bucket. |

## Proposed Tighter Schema

*(Note: this proposal predates the `pmc_nd_form` and `arpa_e` categories above and is retained
as a design sketch — any future consolidation should start from the current 11-category schema.)*

| Label | Combines current labels/cues | Logic | Examples of cues | Rationale for change |
|---|---|---|---|---|
| Grants and Awards | `grant_or_award`, `formula_grant`; optionally FOA-backed awards where project-specific award language appears. | Group explicit grant/award language and formula-allocation grant language; classify FOA cases here only when FOA appears near project-specific award or recipient language. | `grant`, `award`, `grant award`, `formula grant`, `formula-based award`, `EECBG`, `selected to receive`, `recipient of`, `Funding Opportunity Announcement` near `grant` or `award`. | This is the dominant category. Formula grants are still worth retaining as a subtype because they represent a different allocation channel. |
| Loans and Loan Support | `loan_guarantee`, `federal_loan`, `revolving_loan`. | Group debt-related funding mechanisms where the federal role is lending, guaranteeing a loan, or capitalizing a revolving loan fund. | `loan guarantee`, `guaranteed loan`, `federal loan`, `loan from DOE`, `revolving loan`, `Revolving Loan Fund`. | These are all debt-related mechanisms and are easier to understand as one family with subtypes. |
| Cooperative Agreements | `cooperative_agreement`. | Keep explicit cooperative-agreement language separate because it names a specific federal assistance instrument. | `cooperative agreement`, `under the terms of the cooperative agreement`, `cooperative agreement with`, `through a cooperative agreement`. | This is a distinct federal assistance instrument and should stay separate when explicitly detected. |
| Cost-Shared Federal Assistance | `cost_share`, plus projects where cost-share appears alongside grant/award/cooperative-agreement cues. | Treat cost share as a financing attribute whenever the evidence describes a federal/non-federal split, even if another mechanism is the primary instrument. | `cost share`, `cost-shared arrangement`, `federal cost share`, `recipient cost share`, `DOE share`, `non-Federal share`. | Cost share is better treated as an attribute or flag, not a mutually exclusive mechanism. |
| Other Financial Assistance | `financial_assistance`, weak `generic_funding` with affirmative project-specific support language. | Group affirmative funding language where the project clearly receives federal support but the instrument is not named. | `financial assistance`, `DOE would provide funds`, `DOE funding`, `federal funding`, `recipient of federal funding`, `would receive federal funds`. | This captures real federal funding evidence where the instrument is not named. |
| Unresolved / Needs Review | `unknown_funding`, especially blank sidecar evidence or FOA/header-only cases. | Use only when the main classifier says the project is funding-triggered but the detail extractor cannot recover project-specific mechanism evidence. | Blank sidecar evidence; `setfit prob=...` only; `manual_label` only; FOA header only; `American Recovery and Reinvestment Act: [x]` without grant/award/loan language. | This should not be interpreted as a mechanism. It is an extraction-quality bucket. |
