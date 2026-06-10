import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {
        "project_id": "ba1bdc0c-a381-1518-898f-4f334996b05b",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "f3b1b05198c19dd996e5",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "88ea90d0-ab8c-2eb5-8ed0-f1df2240e79a",
        "initiation_candidate_id": "b4889d07540d4778eb49",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "0b74cd9e-1753-d9ef-bc26-31bfc97f6e7e",
        "initiation_candidate_id": "47ebbe1da8a30b4c5df9",
        "decision_candidate_id": "27cf53605910fdc36efb",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "4efde57b-7b2e-986f-ede3-cb2586c43265",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "57026e9b959d60f8fdd9",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "8d263a61-88e8-9799-2426-25857516c0c4",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "7878e120b897ecacc2ba",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "a6d9cef5-c3d0-89c0-3619-ee390735efcc",
        "initiation_candidate_id": "6d8940d8c8fce60c3d03",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "e507acad-d991-d3c7-18e4-a44e69c522ff",
        "initiation_candidate_id": "1f06012cdf99080850e6",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "0454241f-4d72-a4eb-e447-76cacd7b306e",
        "initiation_candidate_id": "6e3639d12534326b2107",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "ade7ea78-89bc-a2c1-9098-8402736a3731",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "45d398d3-090a-ac22-eec5-53182f0790f3",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "3237d7dde498eb5dedc5",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=operative ROD"
    },
    {
        "project_id": "964fea1b-e5ca-0d78-c692-c5cfbbaa232d",
        "initiation_candidate_id": "977eb6e794e981747524",
        "decision_candidate_id": "22331ec8abe80a8aef7c",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "03d82b2a-889d-df48-f5eb-cc1a1a856404",
        "initiation_candidate_id": "27fd812b6cbe9b363e18",
        "decision_candidate_id": "fc559a31d4ee0ce0fe08",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "947ee1c3-f03f-d301-4d22-9669de7733b7",
        "initiation_candidate_id": "92b68baac668743bf0b7",
        "decision_candidate_id": "7df7bba8b8dbe29aa298",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "bcc12bf0-d165-abac-9c70-cae23e33e508",
        "initiation_candidate_id": "afdac9b08a9f3a350825",
        "decision_candidate_id": "66cba37b9779ca2c217d",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "4c208cff-1e25-7e06-c140-b014745630cc",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "cd6d4ba06983f867dc7f",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authoritative register determination"
    },
    {
        "project_id": "afd8711b-d727-e70e-7d42-11b0c50bca4a",
        "initiation_candidate_id": "e3aa8b7ef517708b5366",
        "decision_candidate_id": "2e1769e239ab5c7b1a13",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "d4ad69e9-1f6a-c87f-ec71-3e8a50ec6aa8",
        "initiation_candidate_id": "88dfac6a1ce621ae0d69",
        "decision_candidate_id": "d8ff39b65ea1361f9f36",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "08603fe3-0c36-f30f-3ecf-f6c5fc101608",
        "initiation_candidate_id": "ba7e633a5773aa03d6a5",
        "decision_candidate_id": "5591a75fb147a06c8dd7",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "e508af54-43f8-0225-cdfd-1237555bcea8",
        "initiation_candidate_id": "d5701881e8d0af760bd8",
        "decision_candidate_id": "8ae7f8839e7d0aaaa386",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "b34fdcd6-dbee-ae5a-8fd2-c33e673664fb",
        "initiation_candidate_id": "d9bc51e39da2ecc8f7dd",
        "decision_candidate_id": "73c198de9300c23ee844",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "7018c05a-ff93-40d4-17af-68f63d155c9c",
        "initiation_candidate_id": "24197bbbf6ce4d2b8654",
        "decision_candidate_id": "ef758bee5aa3fc7c36d0",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "475f5ae7-03e4-0545-16e3-cd42e1fe2504",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "723d78660275a94b4b3b",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "c983b48f-16a3-8e90-7ad8-eca4649e890c",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "2f18abffa2ec86b7c8c4",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "385c5771-58bb-6e55-5c91-1b00c7a9aebf",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "aafb9bc069d52bca5407",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "1ab7c626-39a4-c18b-a880-31b6aa84afca",
        "initiation_candidate_id": "6132e500586d41b9fb99",
        "decision_candidate_id": "1213084536fb309dc8ef",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "33e8f084-559a-e08c-d817-2541514b1afa",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "none",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "ed300094-71c0-1593-f108-fc3e68f982c5",
        "initiation_candidate_id": "0ba32c7c992389138c6e",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "1d3ff3cb-7637-2d04-c83e-f8979597767d",
        "initiation_candidate_id": "6a234b91753d05c4856d",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "1f4d1e9c-5481-ca5e-6ec6-e5fae8fb687c",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "3d208c88a86acd3d0234",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "696e35bc-4555-abd6-d760-bbaf9a1675a7",
        "initiation_candidate_id": "b47ff2beff7c3acb72ef",
        "decision_candidate_id": "d2b232d48133279ccabd",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "1d74eaf6-f303-40ff-e76d-a9b80107627a",
        "initiation_candidate_id": "ccb4a0f081b26dac2930",
        "decision_candidate_id": "cc7c1189cb6b44d96a90",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "dd11df9f-80de-9b8d-d5b9-2640c039c7f8",
        "initiation_candidate_id": "0a348c531bcdb0b8e9cb",
        "decision_candidate_id": "70d536c2ab96449dea87",
        "notes": "init=DOE Initiator signature; dec=authoritative register determination"
    },
    {
        "project_id": "f175c040-547d-9696-6c37-0473d3d6ba35",
        "initiation_candidate_id": "1d2ca1fdd5ef2993c87d",
        "decision_candidate_id": "fc640d4f2d02943ee067",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "7fe53e35-a0ee-c15b-b448-4e4cd3643aa9",
        "initiation_candidate_id": "e493d29ad1d3e7b432a3",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "a5deef83-ea92-b63c-68a2-8a876fead29d",
        "initiation_candidate_id": "d3b09fc8ebc39a1a8419",
        "decision_candidate_id": "e0d46e6bd4a4a5e7baa0",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "f6b6d0d3-2589-acca-0463-7e36aa6afd71",
        "initiation_candidate_id": "ed5e1c1629e66a9cf135",
        "decision_candidate_id": "d25c0417a359ce6c0012",
        "notes": "init=earliest qualifying application or scoping start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "21ed0f9c-2422-fdff-3bf3-83cc72dae095",
        "initiation_candidate_id": "04630b320217a3d920fd",
        "decision_candidate_id": "f73a21f6054a898dcab5",
        "notes": "init=earliest qualifying application or scoping start; dec=authoritative register determination"
    },
    {
        "project_id": "210745db-62cc-baa2-2507-7bc19a645a98",
        "initiation_candidate_id": "c849086f6e2057409731",
        "decision_candidate_id": "8275ec88d21289613e84",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "0485f560-aec0-c425-20d8-a4a86f15e78e",
        "initiation_candidate_id": "8d7bdd68a5dccf10cff1",
        "decision_candidate_id": "dc6fc4ca31021bdcc687",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "d3f7ddbb-c4a1-bfed-4332-727a81296d37",
        "initiation_candidate_id": "db33c79d26f803bc13ed",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "11364509-5cdb-ac5d-411a-224e1fb144f1",
        "initiation_candidate_id": "none",
        "decision_candidate_id": "84a8ac59acf6b7285e8b",
        "notes": "init=none; no qualifying NOI, application, scoping, or CE start candidate; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "1295f42a-a5da-e0fb-c9f6-69bcf94879b2",
        "initiation_candidate_id": "5503cd2b8b89c4bcfe3e",
        "decision_candidate_id": "c26cf99cca49cea35f58",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "6c8de8e6-48cc-34bc-f507-50bf9d204e29",
        "initiation_candidate_id": "641e117850853a37b005",
        "decision_candidate_id": "ffdb271f51fd535c78db",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "6b60697a-368f-304d-e71f-d5413b6d19cb",
        "initiation_candidate_id": "cc36e0789fc768b86c6b",
        "decision_candidate_id": "333b61b0ed3611592efe",
        "notes": "init=earlier CE Date Determined paired with later operative signature; dec=authoritative register determination"
    },
    {
        "project_id": "4bc07763-fcaf-85a7-741f-47868e106d3d",
        "initiation_candidate_id": "fd7e50375271d9b67913",
        "decision_candidate_id": "none",
        "notes": "init=earliest qualifying application or scoping start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "2dd0e23c-8101-fabc-a512-813c4750ccfb",
        "initiation_candidate_id": "7b235b44a594beaa865d",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "edcbf356-2521-c8c8-c1d6-3c105903a659",
        "initiation_candidate_id": "5d517462c943a4118c90",
        "decision_candidate_id": "373a9a457bd278efa69a",
        "notes": "init=DOE Initiator signature; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "143a87ef-bb29-6e39-7a41-50365dd83c39",
        "initiation_candidate_id": "7e2f40212bc96edd3316",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
    },
    {
        "project_id": "7d0eb9bd-d4f4-aa72-67ef-47249d6f5f14",
        "initiation_candidate_id": "ce66b4216e4ce21bc460",
        "decision_candidate_id": "202c09da4446f75f068f",
        "notes": "init=authoritative register project start; dec=authorizing-official or compliance-officer signature"
    },
    {
        "project_id": "c3df5436-a345-e050-9c72-0db1c99c2704",
        "initiation_candidate_id": "b3481956eb2387ecb106",
        "decision_candidate_id": "none",
        "notes": "init=authoritative register project start; dec=none; no operative CE determination candidate"
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
