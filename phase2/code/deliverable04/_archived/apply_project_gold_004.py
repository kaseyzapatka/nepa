import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "552e51a3b257f1e8c7d90d1bab4a2ce0",
        "initiation_candidate_id": "70b6f4795dce8eb2da79",
        "decision_candidate_id": "5ac8346b423941a1e1e4",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "5e9a3283f3aac48201f1725f75b46666",
        "initiation_candidate_id": "01c7c8828cdf1afb8f0a",
        "decision_candidate_id": "ad059ef1c211308b31b9",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "fdf78f8672fed04b2765b1db3f1f94d0",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "2ae7504e261e35c9bd9eb95ac637c82a",
        "initiation_candidate_id": "eb7c2582d03924df1136",
        "decision_candidate_id": "5a505fdb544c14bd4636",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "793ee63e4b9621cd9c2d200e1eb81de6",
        "initiation_candidate_id": "33584c5ff268a8138739",
        "decision_candidate_id": "abc83ab922e814c40d47",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "685a0b42113cbea472c150d1624bbd8a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "b52d2bb7537c6bef6286",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "99ef7d6fb0e2daf79abb08148c992e78",
        "initiation_candidate_id": "54f18e16e440c56791a0",
        "decision_candidate_id": "031f1d5370aa941546f0",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "27b8eab66b62bd7fac3bc7f79a470eb1",
        "initiation_candidate_id": "b30f7e6207e46e2cac5c",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "8237e9b86021dba1b116398c74644f23",
        "initiation_candidate_id": "d91c1989c3d597c6d231",
        "decision_candidate_id": "ae3a02ff4178fbcf5161",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "74707768051b207cfb55f02ab4a2e5e7",
        "initiation_candidate_id": "0750680937b98f2980ef",
        "decision_candidate_id": "663cedabf62a326759b9",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "bcbb86822312817d6dfd24cb3a2ab636",
        "initiation_candidate_id": "7ae70565ec94d9234215",
        "decision_candidate_id": "bb9c40a76658b923f43d",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "2ba9fd812367e514dc1f5ff9b667b18d",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "67d5554c8ab413294e40",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "7adcbad739b9ad5409dd7a14acc3bc18",
        "initiation_candidate_id": "80e63f3bb24c3c902e4c",
        "decision_candidate_id": "71357359d5d9d4b90231",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "07e8b0cd7c9e9abf8fb276acae7ce0d0",
        "initiation_candidate_id": "7a934648181584ef286e",
        "decision_candidate_id": "ede2a651f558e86f9b31",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "66c9b0ec3e30e728787f3b9eb502ddfc",
        "initiation_candidate_id": "83c582b589a710db9715",
        "decision_candidate_id": "897b4bbaa122f502e2cb",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "425dc3f36a117689f9229326bbf7c46a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "174adbebbf1247b5d92610efc740f533",
        "initiation_candidate_id": "1497d556f34109af53b6",
        "decision_candidate_id": "db7c5658b496ba0f2d12",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "3002af707d1ddb1294f547e626f784fe",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "54b270258af450833561bb404cd74179",
        "initiation_candidate_id": "b988f8674c137dfd7f17",
        "decision_candidate_id": "1f8d9bf0804200133eee",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "c7dbd66ba2183fb5ea44d5e1772944d8",
        "initiation_candidate_id": "c2f6d71c64c493e17460",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "4ed07a0ef8f6bf55c20005e517bc7579",
        "initiation_candidate_id": "a9e5afa0d4bab9a01a04",
        "decision_candidate_id": "5213d679d77fcc844cd7",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "d7c3aeeb288657ce7ec10f1d3ff72d4e",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "82bb20eb25b7ddc7453703a918b70148",
        "initiation_candidate_id": "3318064845a7a6b1d42f",
        "decision_candidate_id": "7d134819b10dfad07c58",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "84b153f015fab166e70a6a40aa96414d",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "6c2562285a64607ff7d8",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "bb4186770676398ec8143f4da2955ec7",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "dd8606ef271623a6148c059982e43a9d",
        "initiation_candidate_id": "034a40888b85713e77a4",
        "decision_candidate_id": "e711dc8d2986e02ea6f4",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "9e9978cda8624597659d4b965b9eeb2b",
        "initiation_candidate_id": "5d676f975adc3c9b54ec",
        "decision_candidate_id": "b397068ef61471093342",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "43a3ed9ea3cae13472bf9ad81f0f46a6",
        "initiation_candidate_id": "b23382b0ef5a64f29439",
        "decision_candidate_id": "c345561e780240d94307",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "1c43ac38157653b1c7c339ec8598a85b",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "579c78fe6f12f2c72687aecf09223fc6",
        "initiation_candidate_id": "d16ad34bee9ecafa2385",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "626af2e0dcb8b43248228cd876240b6d",
        "initiation_candidate_id": "9c1a90e42dbf89f13d6c",
        "decision_candidate_id": "91cabd1161b39336c3d9",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "9baf106759ce28ac9e7fc53c05d5b646",
        "initiation_candidate_id": "3ccc4f2039cdb964ca0b",
        "decision_candidate_id": "ccbf62a5feeef09e292d",
        "notes": "init=earliest qualifying application or scoping start; dec=operative FONSI"
    },
    {
        "project_id": "a69b8da81ff83ca297a4e6b5cfeca6de",
        "initiation_candidate_id": "9ece6a476113ae70c01e",
        "decision_candidate_id": "2c7b08700902c345ce5f",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "c2df9af69865ab70ed8da4ee63d4045e",
        "initiation_candidate_id": "dd63224ef144531fd0b6",
        "decision_candidate_id": "e77e825bbe6574cab8c5",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "3d61452bf76c4b6d2a6364eb725295c4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "ca2dc1eff43eb74afcc2e1ad006ab370",
        "initiation_candidate_id": "ceb759ec79784df4ec19",
        "decision_candidate_id": "0401742a46b4c210a979",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "d66218d369b7f188037e71c2536a283d",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "9449a3124cdabff9eccc974152960afd",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "f14c3277f8dcc9e9b0431660be5a6717",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "556ce5519c1a0973fd64",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative ROD"
    },
    {
        "project_id": "6544356c54a8e76259ed8f0bbc4a353a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "a32d5a1a71792cf4ac97",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "22b29df8b8f92881a4da801205255279",
        "initiation_candidate_id": "10b12e530989eb86a53b",
        "decision_candidate_id": "77f8004ff12b8b96f262",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "1c20864d3da03b2151e23487fc6bc9dd",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "782e66e9fbb3dd907ea07de28ab7e66b",
        "initiation_candidate_id": "4c4d08b334ec07102bcd",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "6b62e653a15112e363a095c829448789",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "fb84d646d83d033afb61",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "6e3605db977431e23c9cebd30719624d",
        "initiation_candidate_id": "f9c5540d3b7662242c8e",
        "decision_candidate_id": "4029be8bde3c85ab6955",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "b75c50c1ff6f9f8399195eb693d515d7",
        "initiation_candidate_id": "6e4ef7e3442e4c771191",
        "decision_candidate_id": "2faa66fef52c0fed59f7",
        "notes": "init=authoritative register project start; dec=authoritative register determination"
    },
    {
        "project_id": "5ab3dbc04373452c2e02d1c0e762f3c2",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "dc71612c1b5bb4028ce5",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "243c885992cc415a95459b8ba8c8980a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no precise day-level FONSI/ROD candidate"
    },
    {
        "project_id": "4632f0325d001c287445acf3773e6299",
        "initiation_candidate_id": "442508703d45b7d6ff00",
        "decision_candidate_id": "3383868d03978eaaaa3c",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "0f534b747775fd35fc272eb78c71f6ea",
        "initiation_candidate_id": "b37200f310a1b3bad7a8",
        "decision_candidate_id": "25fa8429cff17ab568d7",
        "notes": "init=earliest qualifying application or scoping start; dec=operative ROD"
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
