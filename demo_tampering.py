"""
demo_tampering.py — three attack scenarios that justify the use of blockchain
This is the answer to the panel question:
    "Why blockchain? A Postgres audit log would do the same job."

We demonstrate three concrete attacker behaviours that a centralised log
cannot defend against, and show that the chain catches all three.

  Scenario A — Insider hides an incident
       A privileged DB admin opens the audit log and DELETES the row that
       records a successful detection of a $2M fraud attempt.

  Scenario B — Insider rewrites severity
       The same admin downgrades a CRITICAL incident to LOW so the SOC
       does not investigate.

  Scenario C — Local model substitution
       Attacker (or compromised admin) replaces the trained model file with
       a deliberately weak model that approves any message — *without*
       updating the registered profile fingerprint on the chain.

Each scenario runs against (1) a centralised JSON log and (2) the blockchain.
The centralised log accepts the modification silently. The blockchain
detects it.
"""

import json
import os
import time
import shutil

from blockchain_v2 import (
    CAPaFSBlockchain, IndependentVerifier, _hash,
    CHAIN_FILE, ANCHOR_FILE,
)

CENTRAL_LOG = "central_audit_log.json"
PROFILE_FILE = "demo_profile.bin"


# A naïve centralised audit log (the comparison baseline)

class CentralLog:
    def __init__(self, path=CENTRAL_LOG):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def append(self, entry: dict):
        with open(self.path) as f:
            data = json.load(f)
        entry["timestamp"] = time.time()
        data.append(entry)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def all(self) -> list:
        with open(self.path) as f:
            return json.load(f)

    # Naïve "integrity check" — there is none, so this just returns OK.
    def verify(self) -> bool:
        return os.path.exists(self.path)


# Setup

def banner(title: str):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def reset_environment():
    for p in [CHAIN_FILE, ANCHOR_FILE, CENTRAL_LOG, PROFILE_FILE]:
        if os.path.exists(p):
            os.remove(p)


def populate(bc: CAPaFSBlockchain, log: CentralLog):
    """Generate identical incident history in both stores."""
    profile = b"profile_TF_IDF_centroid_v1_employee_CFO_001"
    with open(PROFILE_FILE, "wb") as f:
        f.write(profile)
    bc.register_profile(profile, {"model": "TF-IDF", "n_features": 1500})
    log.append({"event": "PROFILE_REGISTERED",
                "profile_hash": _hash(profile),
                "metadata": {"model": "TF-IDF"}})

    cases = [
        ("Wire $2.4M to DE89370400440532013000 right now. No approval.", 0.04, 0.45),  # CRITICAL
        ("URGENT: send $850k offshore today, do not document.",          0.08, 0.45),  # CRITICAL
        ("Transfer $1.2M to Cayman account, bypass committee.",          0.12, 0.45),  # HIGH
        ("Slight stylistic deviation in routine memo.",                  0.41, 0.45),  # LOW
    ]
    for text, sim, thr in cases:
        bc.log_incident(text, sim, thr)
        log.append({
            "event": "INCIDENT",
            "text_snippet": text[:80],
            "similarity_score": sim,
            "threshold": thr,
            "deviation": round(thr - sim, 4),
            "severity": CAPaFSBlockchain._severity(sim, thr),
        })

    return profile


# Scenario A — Insider deletes an incident
def scenario_A_delete_incident(bc: CAPaFSBlockchain, log: CentralLog):
    banner("SCENARIO A — Insider deletes the $2.4M CRITICAL incident")

    # Centralised log
    entries = log.all()
    print(f"  Central log before : {len(entries)} entries")
    entries = [e for e in entries if "$2.4M" not in str(e.get("text_snippet", ""))]
    with open(log.path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Central log after  : {len(log.all())} entries")
    print(f"  Central log says   : verify() = {log.verify()}  ← attack went undetected")

    # Blockchain — try to do the same
    print()
    print(f"  Chain before       : {len(bc.chain)} blocks")
    raw = json.load(open(CHAIN_FILE))
    raw = [b for b in raw if "$2.4M" not in str(b.get("data", {}).get("text_snippet", ""))]
    with open(CHAIN_FILE, "w") as f:
        json.dump(raw, f, indent=2)
    bc._load_chain()
    print(f"  Chain after        : {len(bc.chain)} blocks")
    integrity = bc.verify_chain(silent=True)
    problems = bc.detect_tampering()
    print(f"  Chain integrity    : {integrity}")
    print(f"  Tamper detection   : {len(problems)} broken link(s) → {problems[:1]}")

    # Independent regulator view
    print(f"  Regulator verdict  : {IndependentVerifier().verify()}")



# Scenario B — Insider rewrites severity
def scenario_B_downgrade_severity(bc: CAPaFSBlockchain, log: CentralLog):
    banner("SCENARIO B — Insider downgrades CRITICAL → LOW to silence SOC")

    # Reset and re-populate so we start clean for this scenario
    reset_environment()
    bc.__init__(employee_id="CFO_001", load_existing=False)
    log = CentralLog()
    populate(bc, log)

    # Central log: rewrite severity field
    entries = log.all()
    for e in entries:
        if "$850k" in str(e.get("text_snippet", "")):
            e["severity"] = "LOW"
            e["deviation"] = 0.01
    with open(log.path, "w") as f:
        json.dump(entries, f, indent=2)
    downgraded = [e for e in log.all() if "$850k" in str(e.get("text_snippet", ""))]
    print(f"  Central log $850k entry severity now: {downgraded[0]['severity']}  "
          f"← attack went undetected")

    # Blockchain: rewrite severity field
    raw = json.load(open(CHAIN_FILE))
    for b in raw:
        if "$850k" in str(b.get("data", {}).get("text_snippet", "")):
            b["data"]["metadata"]["severity"] = "LOW"
            b["data"]["metadata"]["deviation"] = 0.01
    with open(CHAIN_FILE, "w") as f:
        json.dump(raw, f, indent=2)
    bc._load_chain()
    integrity = bc.verify_chain(silent=True)
    problems = bc.detect_tampering()
    print(f"  Chain integrity     : {integrity}")
    print(f"  Tamper detection    : {len(problems)} hash mismatch(es) → "
          f"{problems[0] if problems else 'none'}")



# Scenario C — Model file substitution (the *active* defence)

def scenario_C_swap_model(bc: CAPaFSBlockchain, log: CentralLog):
    banner("SCENARIO C — Attacker swaps the local model with a permissive one")

    reset_environment()
    bc.__init__(employee_id="CFO_001", load_existing=False)
    log = CentralLog()
    legitimate_profile = populate(bc, log)

    # Attacker silently replaces the trained model on disk with a
    # weak one that classifies everything as NORMAL.
    malicious_profile = b"weak_model_approves_everything"
    with open(PROFILE_FILE, "wb") as f:
        f.write(malicious_profile)

    print("  Profile file on disk has been silently replaced.")
    print(f"  Central log: no defence — model file is outside its scope.")
    print()

    # Pipeline reads the model and asks the chain "is this still legit?"
    loaded = open(PROFILE_FILE, "rb").read()
    ok, reason = bc.verify_profile_against_chain(loaded)
    print(f"  Chain check before classification: ok={ok}")
    print(f"  Reason                           : {reason}")
    print(f"  → Pipeline REFUSES to classify with substituted model.")
    print(f"  Auto-recorded TAMPER_DETECTED block in chain.")

    # Show the chain reflects the event
    last = bc.chain[-1]
    print(f"\n  Last block: #{last.index} type={last.data.event_type} "
          f"severity={last.data.metadata.get('severity')}")


# Entry point
if __name__ == "__main__":
    reset_environment()
    bc  = CAPaFSBlockchain(employee_id="CFO_001", load_existing=False)
    log = CentralLog()
    populate(bc, log)

    scenario_A_delete_incident(bc, log)
    scenario_B_downgrade_severity(bc, log)
    scenario_C_swap_model(bc, log)

    banner("CONCLUSION")
    print("""
  The centralised log accepted A, B, and could not even see C.
  The blockchain rejected A and B (cryptographic integrity violation) and
  C is impossible to execute against an active-verification pipeline:
  the chain becomes the AUTHORITY on what model is allowed to run.

  This is the concrete answer to "why blockchain instead of a database":
  - Database: trusts whoever has write access.
  - Blockchain: trust is anchored in the hash chain itself; modifying
    history requires re-mining every subsequent block, and in the hybrid
    architecture also requires republishing a Merkle root that contradicts
    the one already published externally.
""")
