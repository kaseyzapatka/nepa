import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "dde41f036a9bd2ffd6916c6f8c9bdf41",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "f535bb6a0fca98f2e360",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "af0dc77794f31d272dccbe8e3b57c26e",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "5d955635009eb63142c7b12770cd5bbb",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "c41bd49b08d9f44f40ce",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "4e6cfe21483aa57b3aa2532e6557fa3b",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "be1cf87f42cd80d7aa3b2587d99873c7",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "d7bff39ed4d389feaf72eba5c45f17b6",
        "initiation_candidate_id": "0074cce247e9494d7237",
        "decision_candidate_id": "8b47fb30c8c9fb8a5017",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "5e78bf80baf836f0b097258d2c9e6675",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "9e1cd863145237d2b359",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "6814d5639775c55daf639c5c4f601c56",
        "initiation_candidate_id": "f5d2a1004ab656ce8379",
        "decision_candidate_id": "18bb99291abb590921bc",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "df80cced6a7403556813b21162b20996",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "cb99e74b725606eea8a9",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "71b3becd8598acbf85a16c09ffea5b82",
        "initiation_candidate_id": "d74864a26cc0b1cfe8f7",
        "decision_candidate_id": "aba8db1fe06d8b31032c",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "ed0d144d03473d3e72ba81b1de97c77d",
        "initiation_candidate_id": "c747431617675bb8b22f",
        "decision_candidate_id": "e00641657b361c471fd5",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "eeb42f45c169ddbce9f5cb5ab081e140",
        "initiation_candidate_id": "a79a4c4316f598639ad9",
        "decision_candidate_id": "413c24619cdcda8694b8",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "d209965bb3a54dd7009fc9e057acf5e7",
        "initiation_candidate_id": "f60ad6de084bbebd1fda",
        "decision_candidate_id": "ce5f2158ba58b4dcfd85",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "423b0bf50ca043b7cd2fc5f21b8689bf",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "ce6fe01b18e3fdb1285f",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "47ef0c68cd7f5b875357e28c772e4145",
        "initiation_candidate_id": "a159820621bd4b3bf080",
        "decision_candidate_id": "fb5a8c7e3ee521cba6cd",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "cce33ee127bbe8cbe6c7d3321ce978fe",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "a54c3e460c6da696efc0",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "531c0055aa475d2d41f079107f8d4e27",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "5f5ac74cd546edb22204",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "d4036a014dddb83264d1cb676d4b2b61",
        "initiation_candidate_id": "c8f0b3914464b12857ee",
        "decision_candidate_id": "30566b97fb9811e1a07f",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "201d8ca1ab7783426dec89db97739493",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "67b0d1b5066cfdba6dc0",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "f51b20c1c86180625bc8dddd68e24155",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "67e7692a197f12100663ff033ec82e26",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "ebd282b128924df300d82e2ad95e6212",
        "initiation_candidate_id": "ce36b355587332ea6851",
        "decision_candidate_id": "9fa081e99248034c5bdf",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "6ad9161f2ee067150ab0d5c4356332b3",
        "initiation_candidate_id": "379d44a4037c20129917",
        "decision_candidate_id": "dc0e6eec382016468999",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "a36a6d18c0b8149270ccff9dbefa5600",
        "initiation_candidate_id": "7a64869f68dab5b647ea",
        "decision_candidate_id": "baa12231d69c5099050f",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "d336bfed9f53527e4b9a3caf40c65715",
        "initiation_candidate_id": "6f9ac3ecdc5c7ff725c8",
        "decision_candidate_id": "23e308f63ee61d1d63ef",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "19c56b599fbb0dabca87ccaf6476e31c",
        "initiation_candidate_id": "efaf9c6f4e52312d3433",
        "decision_candidate_id": "1ade42ff5b73d7e0923b",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "edcee6574bcaa5d3a07b552386eb0219",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "55a5aa666f6c9cb80f1f",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "f1ffde2f85792ce490b141c74f9baa41",
        "initiation_candidate_id": "b6a9fef9fee17426aaf7",
        "decision_candidate_id": "14b6f73387aa5791724e",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "0f4534b0b82d8d3b2b589184575f8dda",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "dd391cdcd500bbd23ef3",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "debe659941dc65ed630daab88d5fbf81",
        "initiation_candidate_id": "aa513b9070a3d288e9b6",
        "decision_candidate_id": "8be77ea0cf289ed18e67",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "a9bf633f178f02d579da39661ef6ae41",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "a3beb871f4aaa1bdcf46",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "42fa3a31354e0311b21a78fb3ac90c29",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "1a8f4897dbd628033ce8",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "eb3af99b4fd5c823611b28f620e5ca86",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "aae737cf3c81f4a5ae97e36eb54dc5bd",
        "initiation_candidate_id": "93b88f6ac4c4c05fb3b5",
        "decision_candidate_id": "6a9db2afc0c213419a6c",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "07eb4a9c6579964af3a5a52928cabc7e",
        "initiation_candidate_id": "cd8d3f91d2f1045c1e61",
        "decision_candidate_id": "8d79ea75accd4ea176f8",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "dd57f4c89d7bac04b15392d3d5adc606",
        "initiation_candidate_id": "993684e3a2d90f974127",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "62c1cf4692d281b3228e5cf0d064c86f",
        "initiation_candidate_id": "24568bfdc9376e82bd9b",
        "decision_candidate_id": "680c0b7290923bcf943b",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "323825530f7e229f10a719e2219d2cb3",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "f0ea767a275d8e2d007c",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "23811d81ae09a7da1cf9167043268f6c",
        "initiation_candidate_id": "438da777ab979875bb62",
        "decision_candidate_id": "924926e32c072bb781b5",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "fd376168a2ca237d6b4a1e208b62ea3c",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "769bdbd49a96651d4b6a61945446f3c3",
        "initiation_candidate_id": "d75de941924f9ad1488e",
        "decision_candidate_id": "0787a426150fc497faa2",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "292d487f83fadcd1cef13fe9b55b8b89",
        "initiation_candidate_id": "afc84b4b3ad693fa24ad",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "fc80f2310515dbabce64a4f795b4ab8a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "0ea9e68a3a60198c00e3",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "5ed7f88a37957b8a812a19a8e8c41ffe",
        "initiation_candidate_id": "7e1df44fbbe659a5a6c2",
        "decision_candidate_id": "1c19b68edd4eb49ef4f2",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "dd9a887d60e52e18e4958ae6c38f4e66",
        "initiation_candidate_id": "d20cb4ba3d1ca5e61098",
        "decision_candidate_id": "c566efae380cee043966",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "96b26b5a6c877afe0475790d6625c6f3",
        "initiation_candidate_id": "650e431bd5cbb3011700",
        "decision_candidate_id": "8710eb4e1498acea9bb0",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "4f4f11c9b2b50d2d496e5efb30102fa3",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "c6757593d4322cdb6bd327c512ecc7be",
        "initiation_candidate_id": "86b145ae3b63f55d4230",
        "decision_candidate_id": "be0e47f9c37753fddbbb",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "c5a18865cc2aabbe086f9b6f765d619d",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "e0c526f7197d78b3cca85a46a8591ef9",
        "initiation_candidate_id": "4adc85e61f95bb1123cc",
        "decision_candidate_id": "8de27df15068ae24b3e9",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    }
]


sample = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
df = pd.read_csv(sample, dtype=str, keep_default_na=False)
pk = pd.DataFrame(PICKS)
m = df.merge(pk, on="project_id", how="left", suffixes=("", "_new"))
blank = m["initiation_candidate_id"].astype(str).str.strip().eq("")
has = m["initiation_candidate_id_new"].notna()
apply = blank & has
for col in ["initiation_candidate_id", "decision_candidate_id", "notes"]:
    m.loc[apply, col] = m.loc[apply, f"{col}_new"]
m[df.columns].to_csv(sample, index=False)
print(f"Applied {int(apply.sum())} project picks to {sample.name}")
