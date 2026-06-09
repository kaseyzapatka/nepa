import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "f6c9b436-aa1d-52df-9cbf-780c61dfa17d",
        "initiation_candidate_id": "3d815ba023ebb326fd60",
        "decision_candidate_id": "2006cbc469154e34754d",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "b1cb6f2d-2979-91cf-8b0b-53f5b303f8a8",
        "initiation_candidate_id": "51f049e3c5061593a0c9",
        "decision_candidate_id": "1d9714fffed10b0c0c16",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "850764ec-9263-e38e-a71c-a641da649cdd",
        "initiation_candidate_id": "2013d5af08a69a304886",
        "decision_candidate_id": "23998548ef6238fce497",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "04e886be-e1c9-face-45bb-1c0077727683",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "42eb0eca98b0fec09cc1",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "ac2cb93f-58c7-0c57-0b0b-221ffabdc0ab",
        "initiation_candidate_id": "2ec5ab16b5accdbcdda9",
        "decision_candidate_id": "0c92708ab10905f5aaa5",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "3febc71d-4bfd-979b-5590-95efc0d6cb1c",
        "initiation_candidate_id": "12e200a6d39bf8c9ede5",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "07ed02d2-7cbb-5e3f-e74d-c6929d9cffee",
        "initiation_candidate_id": "d391193d2372b435dd68",
        "decision_candidate_id": "1abfdcdbffd3fe87a2cd",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "5daae2af-7644-1562-db85-d2c532731e4c",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "104382d5eabec1d545e2",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "ad5104db-cea2-bdc6-cefa-8e4dea62eeb2",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "0c609ea3-3048-51c9-1308-01903566b581",
        "initiation_candidate_id": "f4fec67e4413901308e4",
        "decision_candidate_id": "a6364d40316c508999d7",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "3f6c06d3-3c61-1b38-bc07-8ee3317365d7",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "dedb65088ada910fe931",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative FONSI"
    },
    {
        "project_id": "a21d2968-c3fc-0b0c-8837-534e80e344f8",
        "initiation_candidate_id": "db9f363d2886a0d8478e",
        "decision_candidate_id": "3b434a473f42283a71b2",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "dcc26215-8ced-8b8f-4609-805796131708",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "05c4bc0fc5aa54536aed",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "078dd7e0-1966-3ae4-857b-a28b0b388dec",
        "initiation_candidate_id": "59527a81e358980e7a26",
        "decision_candidate_id": "aecd577fc3e9787285b7",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "3e1c0fe3-30f7-17d4-fc58-bb84de7efafc",
        "initiation_candidate_id": "14bd2c264dccc32435ac",
        "decision_candidate_id": "50e48a9dba346afe47a0",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "9ba42867-3185-dde6-71b7-dead154271f6",
        "initiation_candidate_id": "c05c52c67a69f702b8ae",
        "decision_candidate_id": "d599baa42227f66e350e",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "c6982bcc-a660-07fe-21d9-11862d750b10",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "23079c1180eaa07fcbc8",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "2539c34d-33eb-1c60-5b6a-9d30093bbcc4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "d4c006076b6f6d6e2494",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative ROD"
    },
    {
        "project_id": "95f6899e-df5d-6a0a-532e-2d0cad5d1f5d",
        "initiation_candidate_id": "f32e8eff1b85f9f95eaf",
        "decision_candidate_id": "a6ddc425bcbfcc803c4d",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "2c82db23-c88c-7049-ff2f-f2c5347c4152",
        "initiation_candidate_id": "183f9df6914db01f43ef",
        "decision_candidate_id": "0e24fc50629b81c78a05",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "201fbf74-33aa-5323-2adf-6fb351540e25",
        "initiation_candidate_id": "e12547b016e2870dbc88",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "9870e6f4-e256-8766-2fbc-4a46e842a9e9",
        "initiation_candidate_id": "dee26a89d4de1667cd2c",
        "decision_candidate_id": "270e7e2fa96d44057b95",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "92b13086-c82f-26a8-0737-ee7fd32cc4cb",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "e1c8209ea351f6f72566",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "1167dca8-16c2-400b-0163-6a6c6ab9a225",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "781b39c3-3af5-dcd4-6f9c-0355fbf0838e",
        "initiation_candidate_id": "4ddbb0a2900a6d11de85",
        "decision_candidate_id": "0ed7cd9b8b053c004f73",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "205a2672-212a-e498-6edc-dd2bce5fbea5",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "3c027b81221ba1141c2f",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "daf27979-126c-52c3-e902-709842b724c4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "2c1cb592-b1b6-c4eb-60a8-a5c50868591d",
        "initiation_candidate_id": "402562907ffecf3f18d0",
        "decision_candidate_id": "9cc27b005495e43ff4be",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "a4fde60a-a82d-c989-dc9d-3d474c94beb3",
        "initiation_candidate_id": "d8088c78e7d1cbf426bc",
        "decision_candidate_id": "8be1e5b321d927e799a1",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "1496d425-901a-c789-fbc4-e6978c376e15",
        "initiation_candidate_id": "bb8f27bd65c052bd2ab1",
        "decision_candidate_id": "dd4e7eb465d4cb814f76",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "2cefa27d-e6dd-1459-b867-6a20a1b91037",
        "initiation_candidate_id": "694b9dcd61e0380785cd",
        "decision_candidate_id": "d8fecbacd2e9fd97ef15",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "938359d7-0db2-4a95-202f-d5689371f825",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "72e28b2e0f24c00daaf3",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "e580fc0d-b93b-2629-fe58-239234d5f8d0",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "f2479827c7b9185245fe",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "c114d326-50b3-f571-60fc-8f69688f1ced",
        "initiation_candidate_id": "a6fb63f6dbc31624979b",
        "decision_candidate_id": "40a69520cba3e9fa408a",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "035f02d4-95cf-c0da-9a6e-da98e8b5b8ee",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "3d2280c21dee46078595",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "97af6d81-13e2-d0d1-a9c0-fe77eeae9793",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "73ae17dbad8c1a86a6a4",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "9a08db0c-def2-c311-1c26-7035a898dce9",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "36c648696ea628b483ef",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "ff510498-4db6-1a58-6811-90f9518d98f9",
        "initiation_candidate_id": "79ef8141d5611c860652",
        "decision_candidate_id": "0ad93ae7976c3d400a7b",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "f559d60c-6e87-48f2-9169-88d657ea10a6",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "c559d4492d68cf63d35e",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "1591c594-db23-5c1e-dffd-d83bec0e3978",
        "initiation_candidate_id": "4708edb177c8054c824d",
        "decision_candidate_id": "7a8a4bc4eb835d19a8ff",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "39739980-9d85-048d-9e66-f5041b6295b3",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "9deb3bcb-4dd4-4a46-3192-4107bcfae4b0",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "3755250fbbe7b7b90712",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "19ebf385-186e-689b-dc8a-b2048749425b",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "6b9a0ccda0042bffd849",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative ROD"
    },
    {
        "project_id": "87ad384b-46df-a2bb-cae5-b756cac97e08",
        "initiation_candidate_id": "fffa8a0181513e7f3b25",
        "decision_candidate_id": "866a12284b6ecde94af7",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "a627ff27-52da-7e00-c55f-e75237f5a340",
        "initiation_candidate_id": "1d4f71182d600d6bca57",
        "decision_candidate_id": "a3cf855a323c6467c54e",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "dfc2f2b2-5392-0051-c647-61ef75820306",
        "initiation_candidate_id": "204980956165880b9741",
        "decision_candidate_id": "7e54c7f77ca5422df0af",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "4d39f407-3acc-f54f-f1d8-272cd1b0928a",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "435bf351401443d7e161",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "cfc31743-5000-eb26-80ab-47178549391b",
        "initiation_candidate_id": "8d877072b6083b839ca4",
        "decision_candidate_id": "05da036fc0af06e96ec1",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "7bdf3994-a131-254a-f340-824f4ec9bc90",
        "initiation_candidate_id": "e7e3c77c539935d5a820",
        "decision_candidate_id": "8e4773c545432400cd3f",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "602038e2-f111-c898-f05f-8072f275fc2e",
        "initiation_candidate_id": "592e647a3bfad0facfa9",
        "decision_candidate_id": "5026b84fc7b33b241603",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
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
