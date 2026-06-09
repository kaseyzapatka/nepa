import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "9b1b28dcce9693aa5f7823a7f4113eb9",
        "initiation_candidate_id": "3b1c0eef882f3ab0ad13",
        "decision_candidate_id": "6240c7bb784574f8c611",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "4ecbde48914fd2902b8c2ca635c4a6fb",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "646993ae7d2ef18404d069dbb08e8f23",
        "initiation_candidate_id": "0ee2f37875ebb9ae92a8",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "f1583be2829137b9b8e6873107bca1f7",
        "initiation_candidate_id": "e15e9af1e5bb9b09ebd9",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "19fa5940f256367fa2863d06dacda897",
        "initiation_candidate_id": "08939006a658e32ee9fb",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "0ddf61e0158e665d9207e09a05001ba7",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "b82b64efd645c02959f29cb99480e132",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "5e8a9511fc56490096e6683ea35ff61f",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "b2cdd899c4cd20443b8b9d88e983057e",
        "initiation_candidate_id": "c29e2cd461c4faa62263",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "cc643182825ad662a4a22d5efff35171",
        "initiation_candidate_id": "04ac5c0fe6cf68861506",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "35a5389c496c0242990df1037e7eda76",
        "initiation_candidate_id": "3a764e48806c43252d97",
        "decision_candidate_id": "440484c300de520c3a21",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "0407f13a2c3901c562530237469a2416",
        "initiation_candidate_id": "0c0ac4d1aa0d914322f4",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "8a81bf3d4af8201e65aa30a73ca3d32f",
        "initiation_candidate_id": "0aa1176f397e0908909d",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "d663c95e0cbd1c0fae14188bd68dd990",
        "initiation_candidate_id": "979acc9eebdbde331610",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "66939785a75d82f3acd17a57920bd7c2",
        "initiation_candidate_id": "9d12cea81f275ff2f29b",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "10eb172109a10fb3c0d9d1e740318216",
        "initiation_candidate_id": "b11b4200326ae24e07f6",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "6b85d77b45be445614d1085f8025844a",
        "initiation_candidate_id": "d51b563e11af4def50a2",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "de063d5e3eddbaeeef1f0fc802cf255a",
        "initiation_candidate_id": "0946a3b8cc7c8b764685",
        "decision_candidate_id": "aea17d82fca130b9ab6d",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "64776431d6b1d7f5cdd415f36148d8db",
        "initiation_candidate_id": "54cafe07da81e14c71ec",
        "decision_candidate_id": "834e12f5d8c6ef73fcc0",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "eed2be582987107fdb9f82c49c851719",
        "initiation_candidate_id": "7381aaa185fdbd9f4417",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "9d57528f4d1ae19633cb4daffe5ebf95",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "cec9eca73d6e8aae147f817ad56132d8",
        "initiation_candidate_id": "a5d260f81432aaf8b5e2",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "305751f0ceccd00a9782d6af607b0623",
        "initiation_candidate_id": "0ac0e7cf198000e87c38",
        "decision_candidate_id": "410b78dcbd0ac61444c2",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "1fa2ded55d753d837aff24c42d8284b8",
        "initiation_candidate_id": "555354f3ad5f6dc679c2",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "e03284b0ae3ed29a839edf5f481839e7",
        "initiation_candidate_id": "ff2d9c373aa6abfc3d05",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "5e4d9b072edc364f98c153b797c406db",
        "initiation_candidate_id": "19bda5b49bd8d3ce34cb",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "16503be533a24ab6be0f16f0bc52c2f8",
        "initiation_candidate_id": "3c2051bc9d3cd79f0f44",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "96c9b42306f28e3f7ed9bab0ad66b348",
        "initiation_candidate_id": "37dd3c42116b3a24b791",
        "decision_candidate_id": "f65a6391580f071d4865",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "357b9b5129c2e699722ef6bcbd1f43a6",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "67edc2a620b557679531dbfd04fc8b77",
        "initiation_candidate_id": "db451ca75ce6cf374b42",
        "decision_candidate_id": "027079090d13f0f3f470",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "e232dd76050a1a3b78326ff59be4291e",
        "initiation_candidate_id": "7363a11fa67d427ff793",
        "decision_candidate_id": "da8616b6276531fe60ad",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "45111254a59f7d9490a1adcba4baf8a6",
        "initiation_candidate_id": "7155b1a04c874b995efa",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "1547f047a8564a060625448f82ebacb6",
        "initiation_candidate_id": "ad6580786c4cb6bb82bf",
        "decision_candidate_id": "80b0b32da6ef0f61d488",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "a60da55a3914121b0a9f5af98ba034d8",
        "initiation_candidate_id": "4792b429a0be9cd0eff3",
        "decision_candidate_id": "239867e788dab312b3d6",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "05673602f1dd6f43856a6a570b9b1597",
        "initiation_candidate_id": "6c97561d6fd8ab492813",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "f46da32543e4529ea955651880d2431c",
        "initiation_candidate_id": "61a88eacf4451fe6a1f6",
        "decision_candidate_id": "91a3ef638b0651f5e784",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "5176eccf47d4246c00cb2adc251663f9",
        "initiation_candidate_id": "38042298ba28e5ecd356",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "1a1ed674b46a04b2a542bf2d4a3bd146",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "af2b74ffc134fdec04be990b8b6a7640",
        "initiation_candidate_id": "e63854a6d96d418196f5",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "7e06b9016e4080d64b61818b1268edc6",
        "initiation_candidate_id": "24a65710aa12761f8f20",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "0580c736d930dff0881d65adcb44f7c9",
        "initiation_candidate_id": "5c3635e23432b5427d5a",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "e4d622c1e93e2153f5fdcba71c338c68",
        "initiation_candidate_id": "fd912214268a4a413ba1",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "dc455ede5d65a3e76e26d9d0fd230f68",
        "initiation_candidate_id": "e5fb8a1e3cc8f61b28ce",
        "decision_candidate_id": "b65f6c64fa24448ddaff",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "1eb5b1ebfb0093d8a995324eed069bf5",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "89414a09bd48fabe4852293e6642776a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "5284a5886f4f41007e97",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "a8c2a50ea818052a3f63af4b0861a103",
        "initiation_candidate_id": "86ebf9374bb6e2f5d841",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "9fafa3ee7103dc969c13e6322408bfb4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "83b874080c80c46e609947a68892b161",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "45240748668e7cd54675745457e9a739",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "f51e0c08b8a773cdaef9",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "262ee70fdc6c6c3043157be7c3ecc649",
        "initiation_candidate_id": "d401286d4de561310180",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
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
