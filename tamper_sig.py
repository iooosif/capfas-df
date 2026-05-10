# tamper_sig.py — shows that Dilithium2 detects block tampering
import json
from blockchain_v2 import CAPaFSBlockchain

bc = CAPaFSBlockchain(load_existing=True)

# find an incident block and tamper with its similarity_score (the most innocuous field)
with open("capfas_chain.json") as f:
    raw = json.load(f)

for b in raw:
    if b["data"]["event_type"] == "INCIDENT":
        b["data"]["similarity_score"] = 0.99   # pretend everything is fine
        print(f"Tampered block #{b['index']}")
        break

with open("capfas_chain.json", "w") as f:
    json.dump(raw, f)

# Reload and check
bc._load_chain()
pq = bc.verify_pq_signatures()
integrity = bc.verify_chain(silent=True)

print(f"Hash integrity : {integrity}")          # False — hash broken
print(f"PQ sig valid   : {pq['all_valid']}")    # False — signature broken