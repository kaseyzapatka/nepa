import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "1dc252f122386a34a12862c17996b9c1",
        "initiation_candidate_id": "c9d6bbbee06f246f0674",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "65e38b98ff7b5f4f4b3f1f6fca8c1b1f",
        "initiation_candidate_id": "d4d3dc5aaed6688649a1",
        "decision_candidate_id": "360ce500f43b867168be",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "da78e2db9863bda0412b8c3d6387140c",
        "initiation_candidate_id": "d4a81654320f5ef4c752",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "ac05b10d6c3ef664ce1a1b25d1f357c6",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "ba0cf860d1b312f76aee7f6c7dcad171",
        "initiation_candidate_id": "55be920a3f07851dcbdb",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "670fde1548246a76ab084f44574a30fb",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "a7ce48e4b1a2ee88caf66bdc9f81732c",
        "initiation_candidate_id": "50e83bdc858fa279dc3a",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "620e152778ddd80f97e589ee78b24f78",
        "initiation_candidate_id": "384097d9d3a8f3be1f9f",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "77a8d5c8c4e6fe2b86b1779a722885e0",
        "initiation_candidate_id": "c58456eda69d454dc87c",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "e50971b6755d3ac9ba812b6a51b14c94",
        "initiation_candidate_id": "73cbcbfbb2209ddb46d6",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "accb3b10db0cf3c6b02815d386728ad6",
        "initiation_candidate_id": "d3609a313554ad3da31d",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "aa47e1a63dac5e0d0d84ade306dca6c4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "468856b169530378904c01ca3b930761",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "4ebfd32914e7ec50b33ca9f61e15c555",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "2d4e2fbbc9d5ee8e8dfe6b7788b15083",
        "initiation_candidate_id": "5e9c0979e5a6e814142a",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "b611d38e2262b5f783c1244393fdc231",
        "initiation_candidate_id": "277d85ca234a813c26a2",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "553ffcdc65e9096e67c69fb4763aad08",
        "initiation_candidate_id": "5c86c891a5920e15d0bb",
        "decision_candidate_id": "7100d70575991885e468",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "692428c0093b17311195bbfa43d12b45",
        "initiation_candidate_id": "2d8400e48c26e6a65435",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "c29c3c63543e37c9456baa468d32b97b",
        "initiation_candidate_id": "e0a53ed0e8b354249e71",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "11fd5877ac29b56b27a77a417db6d90b",
        "initiation_candidate_id": "fab4a66fcfadb3949152",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "dfe6130a176ecde494418bff7e0cd27d",
        "initiation_candidate_id": "dc34b83ee27478424201",
        "decision_candidate_id": "26ced7373f34c9dd5ab5",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "42678ecc94060ec65ebaaf0ee50fbe86",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "cb367c628b158881c12fc100317951a8",
        "initiation_candidate_id": "707a349767d822614fd7",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "8797d81f390e0eb5cb97c8710494e204",
        "initiation_candidate_id": "70a99d91ca687d6f11c5",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "74a77422d9637a3c1a09330478db13e6",
        "initiation_candidate_id": "e0fcc66ec701abf42c44",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "fe5ac2646d3e417e4665dead27fcd77c",
        "initiation_candidate_id": "b60f317f45e667acc158",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "d66667402910dd84ae89bf519e468241",
        "initiation_candidate_id": "d4dad9e9c4b91924a4dd",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "3b79650884348200e94d28f0791ab513",
        "initiation_candidate_id": "adf3734159325642c879",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "20bb7220f8272ac5b35d2c2e1ada7c63",
        "initiation_candidate_id": "322b0c6c0d60425eeddf",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "5156aebf60860219be8950c02b41aea6",
        "initiation_candidate_id": "4ddd5b5a710b78c4fc2f",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "e9ae8b7c2bd04f7bee8b939ce18afc17",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "eeeae37055c133d265ebd33fd7c6dd56",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "d6f63a8b50ec71977089a04cfd0cf983",
        "initiation_candidate_id": "1e37c86e57bd3bc65d1f",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "2ed8b99fee7953749d1788edd3931ce9",
        "initiation_candidate_id": "65ca36f1ce1ce735508c",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "602a221cc45c13b2062ad9987fe9e124",
        "initiation_candidate_id": "2c7d634bcb548934f1e6",
        "decision_candidate_id": "463cfd3a88bbcc94264a",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "605218925a51a7ceb2ca9206cc0cf5b1",
        "initiation_candidate_id": "71c7e5810a6500b28fd2",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "ef3ff314718e334cb84c5451f30323b6",
        "initiation_candidate_id": "68e6e46f0d55181f9a16",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "295200a8ce6a5d352fdde0186bdb6cd1",
        "initiation_candidate_id": "e004f0b2375f806aa8bc",
        "decision_candidate_id": "780e76f2711ee8cde6f4",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "1c0920ceec1cd675cbcbb3ce04cd1af3",
        "initiation_candidate_id": "0c2ffab21e7e4ad34acf",
        "decision_candidate_id": "d10f4939146d46773a21",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "a9931f0087188580c0378bac644abca4",
        "initiation_candidate_id": "588f5114dceadf5a13fb",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "c178fd3d723b2294854d20cd62a8ae36",
        "initiation_candidate_id": "07046bb13703deca407f",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "28115f2c03cc08082f009b340ab9ed34",
        "initiation_candidate_id": "615daed817566bd6f4b4",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "d0786446fe106088227f3366e6a00c2b",
        "initiation_candidate_id": "0717da502d9346c68b41",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "42fa9f318a1fd2a8205ca3665b73de65",
        "initiation_candidate_id": "0a48a0685265dbced7c3",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "a740142c21d8d0439cf0a9df97f2d8b2",
        "initiation_candidate_id": "7cd3c7cf77e1764227b6",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "aae0f3cb01c6c344a90bd05b4c6e1ac2",
        "initiation_candidate_id": "1ede3d745d14419d4f89",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "7c8cde84f2aae499bfc9989d6a019483",
        "initiation_candidate_id": "385306f3b3afba5de830",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "4e63c8e146852bfaaf4d5b6d30bc6c34",
        "initiation_candidate_id": "5ea9adc0828be308d2cc",
        "decision_candidate_id": "9e7f85dcffaba9440e63",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
    },
    {
        "project_id": "926d88776fff8363fe4bd24bed6abc3e",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "38826951e33a7cddc2fc54219d0fe011",
        "initiation_candidate_id": "65a39e24369387c015da",
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
