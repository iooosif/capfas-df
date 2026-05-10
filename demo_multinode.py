"""
demo_multinode.py — multi-node consensus simulation (PBFT-flavoured)
Models the consortium described in the thesis section: BANK + REGULATOR +
AUDITOR + SOC. All four nodes hold an independent copy of the ledger.
A block is committed iff a quorum (>50%) of nodes accept it.

In a real permissioned chain (Hyperledger, Quorum) the genesis block is part
of the network bootstrap — shared, not generated independently per node. We
replicate that here by minting a single genesis and replaying it to every node.

Two scenarios:
  1) Honest network — three legitimate proposals from BANK and SOC.
  2) Byzantine attack — corrupted REGULATOR proposes a forged
     PROFILE_UPDATED block pointing to an attacker-controlled profile.
     Honest nodes apply policy validation (role-based authorisation) and
     reject. Consensus holds.
"""

import time
from copy import deepcopy

from blockchain_v2 import Block, BlockData, _hash, DIFFICULTY, EVENT_TYPES


ALLOWED_PROPOSERS = {
    "PROFILE_REGISTERED":  {"BANK"},
    "PROFILE_UPDATED":     {"BANK"},
    "INCIDENT":            {"BANK", "SOC"},
    "TRANSACTION_BLOCKED": {"BANK", "SOC"},
    "SECURITY_ALERT":      {"BANK", "SOC"},
    "MERKLE_ANCHOR":       {"BANK"},
    "PROFILE_VERIFIED":    {"BANK", "SOC", "AUDIT", "REG"},
    "TAMPER_DETECTED":     {"BANK", "SOC", "AUDIT", "REG"},
}

class Node:
    def __init__(self, node_id: str, role: str, byzantine: bool = False):
        self.id        = node_id
        self.role      = role
        self.byzantine = byzantine
        self.chain: list[Block] = []

    def adopt_genesis(self, genesis: Block):
        self.chain = [deepcopy(genesis)]

    def validate(self, candidate: Block, proposer_id: str) -> tuple[bool, str]:
        prev = self.chain[-1]
        if candidate.previous_hash != prev.hash:
            return False, "previous_hash != local tip"
        if candidate.index != prev.index + 1:
            return False, f"index gap (expected {prev.index + 1})"
        if candidate.hash != candidate._compute_hash():
            return False, "stored hash differs from recomputed hash"
        if not candidate.hash.startswith("0" * DIFFICULTY):
            return False, "PoW invalid"
        if candidate.data.event_type not in EVENT_TYPES:
            return False, f"unknown event_type"
        allowed = ALLOWED_PROPOSERS.get(candidate.data.event_type, set())
        if proposer_id not in allowed:
            return False, f"proposer {proposer_id} not authorised for {candidate.data.event_type}"
        return True, "OK"

    def commit(self, block: Block):
        self.chain.append(deepcopy(block))



class Network:
    def __init__(self, nodes: list[Node]):
        self.nodes  = nodes
        self.quorum = len(nodes) // 2 + 1

    def bootstrap(self):
        genesis = Block(
            index=0, timestamp=1700000000.0,
            data=BlockData(event_type="GENESIS", employee_id="CFO_001",
                           metadata={"system": "CAPaFS-DF", "version": "2.0"}),
            previous_hash="0" * 64,
        )
        genesis.mine(DIFFICULTY)
        for n in self.nodes:
            n.adopt_genesis(genesis)
        return genesis

    def propose(self, proposer: Node, data: BlockData) -> tuple[bool, dict]:
        prev = proposer.chain[-1]
        candidate = Block(
            index=prev.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=prev.hash,
        )
        candidate.mine(DIFFICULTY)
        votes = {n.id: n.validate(candidate, proposer.id) for n in self.nodes}
        accepted  = sum(1 for ok, _ in votes.values() if ok)
        committed = accepted >= self.quorum
        if committed:
            for n in self.nodes:
                ok, _ = votes[n.id]
                if ok:
                    n.commit(candidate)
        return committed, {
            "accepted_by": accepted,
            "quorum_required": self.quorum,
            "votes": votes,
            "block_index": candidate.index,
        }

    def divergence(self) -> dict:
        tips    = {n.id: n.chain[-1].hash[:16] + "…" for n in self.nodes}
        lengths = {n.id: len(n.chain) for n in self.nodes}
        return {"tips": tips, "lengths": lengths,
                "consensus": len(set(tips.values())) == 1}


def banner(t):
    print("\n" + "═" * 72)
    print(f"  {t}")
    print("═" * 72)



def scenario_honest():
    banner("SCENARIO 1 — HONEST CONSORTIUM (BANK / REG / AUDIT / SOC)")
    net = Network([Node("BANK", "operator"), Node("REG", "regulator"),
                   Node("AUDIT", "auditor"), Node("SOC", "security")])
    net.bootstrap()
    bank, reg, audit, soc = net.nodes
    print(f"  Quorum required: {net.quorum} of {len(net.nodes)} nodes")

    ok, info = net.propose(bank, BlockData(
        event_type="PROFILE_REGISTERED", employee_id="CFO_001",
        profile_hash=_hash(b"profile_v1"),
        metadata={"model": "TF-IDF", "n_train": 2000}))
    print(f"\n  1) BANK proposes PROFILE_REGISTERED → committed={ok} "
          f"({info['accepted_by']}/{len(net.nodes)})")

    ok, info = net.propose(soc, BlockData(
        event_type="INCIDENT", employee_id="CFO_001",
        similarity_score=0.05, threshold=0.45,
        text_snippet="Wire $2M urgent...",
        metadata={"severity": "CRITICAL", "incident_number": 1}))
    print(f"  2) SOC  proposes INCIDENT (CRITICAL)   → committed={ok} "
          f"({info['accepted_by']}/{len(net.nodes)})")

    ok, info = net.propose(soc, BlockData(
        event_type="TRANSACTION_BLOCKED", employee_id="CFO_001",
        triggering_block=2,
        metadata={"severity": "CRITICAL", "auto_response": True}))
    print(f"  3) SOC  proposes TRANSACTION_BLOCKED   → committed={ok} "
          f"({info['accepted_by']}/{len(net.nodes)})")

    div = net.divergence()
    print(f"\n  Chain lengths: {div['lengths']}")
    print(f"  Tip hashes   : {div['tips']}")
    print(f"  Consensus    : {div['consensus']}  ← all four nodes agree")



def scenario_byzantine():
    banner("SCENARIO 2 — BYZANTINE ATTACK (compromised REG node)")
    net = Network([Node("BANK", "operator"),
                   Node("REG", "regulator", byzantine=True),
                   Node("AUDIT", "auditor"), Node("SOC", "security")])
    net.bootstrap()
    bank, reg, audit, soc = net.nodes
    print(f"  Quorum required: {net.quorum} of {len(net.nodes)} nodes (1 Byzantine)")

    ok, info = net.propose(bank, BlockData(
        event_type="PROFILE_REGISTERED", employee_id="CFO_001",
        profile_hash=_hash(b"profile_v1"), metadata={"model": "TF-IDF"}))
    print(f"\n  1) BANK proposes legitimate registration  → committed={ok} "
          f"({info['accepted_by']}/{len(net.nodes)})")

    ok, info = net.propose(reg, BlockData(
        event_type="PROFILE_UPDATED", employee_id="CFO_001",
        profile_hash=_hash(b"ATTACKER_CONTROLLED_WEAK_PROFILE"),
        metadata={"reason": "routine update",
                  "attack_note": "silent profile swap"}))
    print(f"\n  2) REG (Byzantine) proposes silent PROFILE_UPDATED → committed={ok}")
    print(f"     accepted_by={info['accepted_by']}/{len(net.nodes)} (need {net.quorum})")
    for nid, (vote, why) in info["votes"].items():
        print(f"       {nid:<5} {'ACCEPT' if vote else 'REJECT':<6}  {why}")

    ok, info = net.propose(reg, BlockData(
        event_type="INCIDENT", employee_id="CFO_001",
        similarity_score=0.5, threshold=0.45,
        text_snippet="fabricated event to dilute SOC alert queue",
        metadata={"severity": "LOW"}))
    print(f"\n  3) REG (Byzantine) proposes fabricated INCIDENT    → committed={ok}")
    print(f"     accepted_by={info['accepted_by']}/{len(net.nodes)} (need {net.quorum})")
    for nid, (vote, why) in info["votes"].items():
        print(f"       {nid:<5} {'ACCEPT' if vote else 'REJECT':<6}  {why}")

    div = net.divergence()
    print(f"\n  Chain lengths: {div['lengths']}")
    print(f"  Tip hashes   : {div['tips']}")
    print(f"  Consensus    : {div['consensus']}  ← honest majority blocks the attack")


if __name__ == "__main__":
    scenario_honest()
    scenario_byzantine()
