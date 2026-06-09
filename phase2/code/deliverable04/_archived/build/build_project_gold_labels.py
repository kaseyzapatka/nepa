import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
CANDIDATES = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
CHUNK_SIZE = 50

INIT_OVERRIDES = {
    "8d263a61-88e8-9799-2426-25857516c0c4": "none",
    "c6982bcc-a660-07fe-21d9-11862d750b10": "none",
    "ba1bdc0c-a381-1518-898f-4f334996b05b": "none",
    "45d398d3-090a-ac22-eec5-53182f0790f3": "none",
    "2c1cb592-b1b6-c4eb-60a8-a5c50868591d": "402562907ffecf3f18d0",
    "4d39f407-3acc-f54f-f1d8-272cd1b0928a": "none",
    "e580fc0d-b93b-2629-fe58-239234d5f8d0": "none",
    "07eb4a9c6579964af3a5a52928cabc7e": "cd8d3f91d2f1045c1e61",
    "292d487f83fadcd1cef13fe9b55b8b89": "afc84b4b3ad693fa24ad",
    "2ae7504e261e35c9bd9eb95ac637c82a": "eb7c2582d03924df1136",
    "423b0bf50ca043b7cd2fc5f21b8689bf": "none",
    "5d955635009eb63142c7b12770cd5bbb": "none",
    "9e9978cda8624597659d4b965b9eeb2b": "5d676f975adc3c9b54ec",
    "d66218d369b7f188037e71c2536a283d": "none",
    "f14c3277f8dcc9e9b0431660be5a6717": "none",
    "28115f2c03cc08082f009b340ab9ed34": "615daed817566bd6f4b4",
    "35a5389c496c0242990df1037e7eda76": "3a764e48806c43252d97",
    "4ebfd32914e7ec50b33ca9f61e15c555": "none",
    "5e4d9b072edc364f98c153b797c406db": "19bda5b49bd8d3ce34cb",
    "66939785a75d82f3acd17a57920bd7c2": "9d12cea81f275ff2f29b",
    "9b1b28dcce9693aa5f7823a7f4113eb9": "3b1c0eef882f3ab0ad13",
    "9d57528f4d1ae19633cb4daffe5ebf95": "none",
    "a7ce48e4b1a2ee88caf66bdc9f81732c": "50e83bdc858fa279dc3a",
    "cc643182825ad662a4a22d5efff35171": "04ac5c0fe6cf68861506",
    "cec9eca73d6e8aae147f817ad56132d8": "a5d260f81432aaf8b5e2",
    "d66667402910dd84ae89bf519e468241": "d4dad9e9c4b91924a4dd",
    "dfe6130a176ecde494418bff7e0cd27d": "dc34b83ee27478424201",
    "eed2be582987107fdb9f82c49c851719": "7381aaa185fdbd9f4417",
    "f46da32543e4529ea955651880d2431c": "61a88eacf4451fe6a1f6",
    "1dc252f122386a34a12862c17996b9c1": "c9d6bbbee06f246f0674",
    "2d4e2fbbc9d5ee8e8dfe6b7788b15083": "5e9c0979e5a6e814142a",
    "605218925a51a7ceb2ca9206cc0cf5b1": "71c7e5810a6500b28fd2",
    "646993ae7d2ef18404d069dbb08e8f23": "0ee2f37875ebb9ae92a8",
    "da78e2db9863bda0412b8c3d6387140c": "d4a81654320f5ef4c752",
    "aae0f3cb01c6c344a90bd05b4c6e1ac2": "none",
    "d336bfed9f53527e4b9a3caf40c65715": "none",
}

DEC_OVERRIDES = {
    "201fbf74-33aa-5323-2adf-6fb351540e25": "none",
    "ed300094-71c0-1593-f108-fc3e68f982c5": "none",
    "1d3ff3cb-7637-2d04-c83e-f8979597767d": "none",
    "4bc07763-fcaf-85a7-741f-47868e106d3d": "none",
    "ad5104db-cea2-bdc6-cefa-8e4dea62eeb2": "none",
    "ade7ea78-89bc-a2c1-9098-8402736a3731": "none",
    "d3f7ddbb-c4a1-bfed-4332-727a81296d37": "none",
    "daf27979-126c-52c3-e902-709842b724c4": "none",
    "eb3af99b4fd5c823611b28f620e5ca86": "none",
    "bb4186770676398ec8143f4da2955ec7": "none",
    "fd376168a2ca237d6b4a1e208b62ea3c": "none",
    "67e7692a197f12100663ff033ec82e26": "none",
    "9e9978cda8624597659d4b965b9eeb2b": "b397068ef61471093342",
    "ca2dc1eff43eb74afcc2e1ad006ab370": "0401742a46b4c210a979",
    "d663c95e0cbd1c0fae14188bd68dd990": "none",
    "6b85d77b45be445614d1085f8025844a": "none",
    "5e4d9b072edc364f98c153b797c406db": "none",
    "05673602f1dd6f43856a6a570b9b1597": "none",
    "7e06b9016e4080d64b61818b1268edc6": "none",
    "e4d622c1e93e2153f5fdcba71c338c68": "none",
    "da78e2db9863bda0412b8c3d6387140c": "none",
    "620e152778ddd80f97e589ee78b24f78": "none",
    "468856b169530378904c01ca3b930761": "none",
    "2d4e2fbbc9d5ee8e8dfe6b7788b15083": "none",
    "b611d38e2262b5f783c1244393fdc231": "none",
    "692428c0093b17311195bbfa43d12b45": "none",
    "3b79650884348200e94d28f0791ab513": "none",
    "b2cdd899c4cd20443b8b9d88e983057e": "none",
    "10eb172109a10fb3c0d9d1e740318216": "none",
    "45111254a59f7d9490a1adcba4baf8a6": "none",
    "af2b74ffc134fdec04be990b8b6a7640": "none",
    "cb367c628b158881c12fc100317951a8": "none",
    "1dc252f122386a34a12862c17996b9c1": "none",
    "a7ce48e4b1a2ee88caf66bdc9f81732c": "none",
    "cc643182825ad662a4a22d5efff35171": "none",
    "66939785a75d82f3acd17a57920bd7c2": "none",
    "cec9eca73d6e8aae147f817ad56132d8": "none",
    "d66667402910dd84ae89bf519e468241": "none",
    "e50971b6755d3ac9ba812b6a51b14c94": "none",
    "accb3b10db0cf3c6b02815d386728ad6": "none",
    "aa47e1a63dac5e0d0d84ade306dca6c4": "none",
    "fe5ac2646d3e417e4665dead27fcd77c": "none",
    "20bb7220f8272ac5b35d2c2e1ada7c63": "none",
    "5156aebf60860219be8950c02b41aea6": "none",
    "e9ae8b7c2bd04f7bee8b939ce18afc17": "none",
    "d6f63a8b50ec71977089a04cfd0cf983": "none",
    "d0786446fe106088227f3366e6a00c2b": "none",
    "c29c3c63543e37c9456baa468d32b97b": "none",
    "c178fd3d723b2294854d20cd62a8ae36": "none",
    "42fa9f318a1fd2a8205ca3665b73de65": "none",
    "ba0cf860d1b312f76aee7f6c7dcad171": "none",
    "38826951e33a7cddc2fc54219d0fe011": "none",
    "646993ae7d2ef18404d069dbb08e8f23": "none",
}


def clean(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def has_flag(row, pattern):
    return bool(re.search(pattern, clean(row.get("positive_cue_flags")), re.I))


def parsed_date(row):
    return pd.to_datetime(row.get("parsed_date"), errors="coerce")


def marked_window(row, before=180, after=240):
    context = " ".join(clean(row.get("model_context")).split())
    marker = context.find("[[")
    if marker < 0:
        return context[: before + after]
    return context[max(0, marker - before) : marker + after]


def semantic_initiation(row):
    window = marked_window(row)
    return bool(
        re.search(
            r"notice of intent|\bNOI\b|project start date|date determined|"
            r"initiator.{0,80}(?:date|signature)|"
            r"application.{0,100}(?:received|submitted|filed|accepted)|"
            r"(?:received|submitted|filed).{0,70}(?:application|request)|"
            r"scoping.{0,90}(?:began|begin|initiated|opened|commenced|start)|"
            r"(?:initiated|began|started).{0,70}(?:scoping|public involvement|"
            r"NEPA process)|scoping (?:letter|notice|package)|"
            r"(?:mailed|posted|distributed|issued).{0,80}(?:scoping|notice)",
            window,
            re.I,
        )
    )


def semantic_decision(row, process_type):
    window = marked_window(row, before=220, after=300)
    negative = bool(
        re.search(
            r"notice of intent|\bNOI\b|public notice of application|"
            r"comment (?:period|deadline|letter)|comments? due|"
            r"public (?:meeting|hearing)|biological opinion|consultation|"
            r"application.{0,80}(?:received|submitted|filed)|"
            r"(?:received|submitted|filed).{0,70}application",
            window,
            re.I,
        )
    )
    positive = bool(
        re.search(
            r"finding of no significant impact|\bFONSI\b|record of decision|"
            r"\bROD\b|decision record|proposed decision|final decision|"
            r"(?:signature|signed|approved|approval|authorized|"
            r"determination|decision date).{0,100}(?:\[\[|date)|"
            r"(?:authorized official|field manager|compliance officer).{0,130}"
            r"(?:\[\[|date)",
            window,
            re.I,
        )
    )
    if negative and not positive:
        return False
    if positive:
        return True
    return process_type == "CE" and bool(
        re.search(
            r"categorical exclusion|decision record",
            window,
            re.I,
        )
    )


def valid_initiation(row):
    role = clean(row.get("candidate_role"))
    if role == "clear_initiation" and has_flag(
        row,
        r"blm_register_tier_a|doe_register_tier_a|fr_noi_metadata|"
        r"doe_initiator_signature",
    ):
        return True
    return role not in {"historical", "reject"} and semantic_initiation(row)


def valid_decision(row, process_type):
    role = clean(row.get("candidate_role"))
    granularity = clean(row.get("date_granularity"))
    if process_type == "CE":
        if granularity not in {"day", "month"}:
            return False
    elif granularity != "day":
        return False

    return role not in {"historical", "reject"} and semantic_decision(
        row, process_type
    )


def apply_override(group, project_id, overrides, current):
    candidate_id = overrides.get(project_id)
    if candidate_id is None:
        return current
    if candidate_id == "none":
        return None
    matches = group[group["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(
            f"Override {candidate_id} for {project_id} matched {len(matches)} rows"
        )
    return matches.iloc[0]


def choose_initiation(group, decision_row):
    candidates = group[group.apply(valid_initiation, axis=1)].copy()

    if clean(group["process_type"].iloc[0]) == "CE" and decision_row is not None:
        decision_date = parsed_date(decision_row)
        paired = group[
            group.apply(lambda r: has_flag(r, r"date_determined"), axis=1)
        ].copy()
        paired = paired[
            paired.apply(
                lambda r: pd.notna(parsed_date(r))
                and pd.notna(decision_date)
                and parsed_date(r) < decision_date,
                axis=1,
            )
        ]
        candidates = pd.concat([candidates, paired], ignore_index=False)

    if candidates.empty:
        return None
    candidates = candidates.drop_duplicates("candidate_id")
    candidates["_date"] = candidates.apply(parsed_date, axis=1)
    candidates["_p"] = pd.to_numeric(
        candidates.get("p_init_cal"), errors="coerce"
    ).fillna(0)
    candidates["_metadata"] = candidates.apply(
        lambda r: int(
            has_flag(
                r,
                r"blm_register_tier_a|doe_register_tier_a|fr_noi_metadata|"
                r"doe_initiator_signature",
            )
        ),
        axis=1,
    )
    candidates = candidates.sort_values(
        ["_date", "_metadata", "_p", "role_confidence_score"],
        ascending=[True, False, False, False],
    )
    return candidates.iloc[0]


def choose_decision(group):
    process_type = clean(group["process_type"].iloc[0])
    candidates = group[
        group.apply(lambda r: valid_decision(r, process_type), axis=1)
    ].copy()
    if candidates.empty:
        return None

    selected = candidates[
        candidates["selected_for_decision"].fillna(False).astype(bool)
    ]
    if not selected.empty:
        return selected.sort_values(
            ["role_confidence_score", "p_dec_cal"], ascending=False
        ).iloc[0]

    candidates["_metadata"] = candidates.apply(
        lambda r: int(
            has_flag(
                r,
                r"blm_register_tier_a|doe_register_tier_a|doe_cx_register_tier_a",
            )
        ),
        axis=1,
    )
    candidates["_p"] = pd.to_numeric(
        candidates.get("p_dec_cal"), errors="coerce"
    ).fillna(0)
    candidates["_date"] = candidates.apply(parsed_date, axis=1)
    candidates = candidates.sort_values(
        ["_metadata", "_p", "role_confidence_score", "_date"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0]


def initiation_reason(row):
    if row is None:
        return "init=none; no qualifying NOI, application, scoping, or CE start candidate"
    flags = clean(row.get("positive_cue_flags"))
    if "date_determined" in flags:
        return "init=earlier CE Date Determined paired with later operative signature"
    if "doe_initiator_signature" in flags:
        return "init=DOE Initiator signature"
    if "register_tier_a" in flags:
        return "init=authoritative register project start"
    if "noi" in flags.lower():
        return "init=NOI publication"
    return "init=earliest qualifying application or scoping start"


def decision_reason(row, process_type):
    if row is None:
        if process_type in {"EA", "EIS"}:
            return "dec=none; no precise day-level FONSI/ROD candidate"
        return "dec=none; no operative CE determination candidate"
    flags = clean(row.get("positive_cue_flags"))
    context = clean(row.get("model_context")).lower()
    if "register_tier_a" in flags:
        return "dec=authoritative register determination"
    if "finding of no significant impact" in context or "fonsi" in context:
        return "dec=operative FONSI"
    if "record of decision" in context or re.search(r"\brod\b", context):
        return "dec=operative ROD"
    return "dec=authorizing-official or compliance-officer signature"


def write_apply(picks, chunk_number):
    out = (
        ROOT
        / f"phase2/output/deliverable04/apply_project_gold_{chunk_number:03d}.py"
    )
    text = f"""import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = {json.dumps(picks, indent=4)}


sample = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
df = pd.read_csv(sample, dtype=str, keep_default_na=False)
pk = pd.DataFrame(PICKS)
m = df.merge(pk, on="project_id", how="left", suffixes=("", "_new"))
blank = m["initiation_candidate_id"].astype(str).str.strip().eq("")
has = m["initiation_candidate_id_new"].notna()
apply = blank & has
for col in ["initiation_candidate_id", "decision_candidate_id", "notes"]:
    m.loc[apply, col] = m.loc[apply, f"{{col}}_new"]
m[df.columns].to_csv(sample, index=False)
print(f"Applied {{int(apply.sum())}} project picks to {{sample.name}}")
"""
    out.write_text(text)
    print(f"Wrote {len(picks)} picks to {out}")


def main():
    sample = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)
    cand = pd.read_parquet(CANDIDATES)
    cand = cand[cand["project_id"].isin(sample["project_id"])].copy()
    for col in ["p_init_cal", "p_dec_cal", "role_confidence_score"]:
        cand[col] = pd.to_numeric(cand.get(col), errors="coerce").fillna(0)

    picks = []
    audit = []
    for sample_row in sample.itertuples(index=False):
        group = cand[cand["project_id"] == sample_row.project_id].copy()
        decision = choose_decision(group)
        decision = apply_override(
            group, sample_row.project_id, DEC_OVERRIDES, decision
        )
        initiation = choose_initiation(group, decision)
        initiation = apply_override(
            group, sample_row.project_id, INIT_OVERRIDES, initiation
        )
        init_id = "none" if initiation is None else initiation["candidate_id"]
        dec_id = "none" if decision is None else decision["candidate_id"]
        notes = (
            f"{initiation_reason(initiation)}; "
            f"{decision_reason(decision, sample_row.process_type)}"
        )
        picks.append(
            {
                "project_id": sample_row.project_id,
                "initiation_candidate_id": init_id,
                "decision_candidate_id": dec_id,
                "notes": notes,
            }
        )
        audit.append(
            {
                "project_id": sample_row.project_id,
                "process_type": sample_row.process_type,
                "initiation_candidate_id": init_id,
                "decision_candidate_id": dec_id,
                "initiation_date": (
                    "" if initiation is None else clean(initiation["parsed_date"])
                ),
                "decision_date": (
                    "" if decision is None else clean(decision["parsed_date"])
                ),
                "initiation_role": (
                    "" if initiation is None else clean(initiation["candidate_role"])
                ),
                "decision_role": (
                    "" if decision is None else clean(decision["candidate_role"])
                ),
                "initiation_p": (
                    0 if initiation is None else float(initiation["p_init_cal"])
                ),
                "decision_p": (
                    0 if decision is None else float(decision["p_dec_cal"])
                ),
                "notes": notes,
            }
        )

    audit_df = pd.DataFrame(audit)
    audit_path = (
        ROOT / "phase2/output/deliverable04/project_gold_proposal_audit.csv"
    )
    audit_df.to_csv(audit_path, index=False)
    print(
        audit_df.groupby("process_type")[
            ["initiation_candidate_id", "decision_candidate_id"]
        ]
        .agg(lambda s: int(s.ne("none").sum()))
        .to_string()
    )
    print(f"Wrote audit to {audit_path}")

    for chunk_number, start in enumerate(
        range(0, len(picks), CHUNK_SIZE), start=1
    ):
        write_apply(picks[start : start + CHUNK_SIZE], chunk_number)


if __name__ == "__main__":
    main()
