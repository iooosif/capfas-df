"""

  1. PERSISTENT chain across runs (the old version wiped the ledger on each
     execution — which defeats the whole point of immutability).

  2. ACTIVE profile-integrity verification: before every classification round,
     the pipeline must call `verify_profile_against_chain()`. If the local
     model file has been tampered with, classification is REFUSED. The chain
     is no longer write-only — it now governs whether the detector is allowed
     to run at all.

  3. Smart-contract-style automated response (chaincode simulation): when an
     INCIDENT block of severity HIGH/CRITICAL is committed, the chain
     automatically appends a TRANSACTION_BLOCKED block and a SECURITY_ALERT
     block, with no manual call. This mirrors Hyperledger Fabric chaincode.

  4. Explicit tamper-detection API (`detect_tampering()`) used by demo scripts
     to show the panel exactly *what* blockchain protects against.

  5. Post-quantum hashing toggle: `HASH_ALGORITHM` in {"sha256","sha3_256"}.
     A single config switch migrates the ledger to a NIST-PQC-aligned hash
     (SHA-3 / Keccak), demonstrating that the hashing layer is interchangeable
     — exactly the migration path described in the thesis.

  6. Merkle-root anchoring: every N blocks, a Merkle root is computed and
     written to `external_anchor.json`. This simulates the hybrid architecture
     (private chain + public Ethereum anchor) discussed in the thesis.

  7. Independent verifier API (`IndependentVerifier`) — a third party can
     verify chain integrity AND match it against an external anchor without
     access to the bank's internal model.

  8. Quantitative self-evaluation (`benchmark()`) — measures storage cost,
     hashing cost, verification cost. Required for the "evaluate your
     solution" part of the assignment.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

# Post-quantum signatures (Dilithium2) and key encapsulation (Kyber512)
try:
    from pq_crypto import Dilithium2, Kyber512
    _PQ_AVAILABLE = True
except ImportError:
    _PQ_AVAILABLE = False



# 0. CONFIGURATION

HASH_ALGORITHM  = "sha3_256"       # "sha256" (classic) | "sha3_256" (post-quantum)
DIFFICULTY      = 2
ANCHOR_INTERVAL = 5
CHAIN_FILE      = "capfas_chain.json"
ANCHOR_FILE     = "external_anchor.json"
PQ_SIGNING      = True             # sign every block with Dilithium2
NODE_KEY_FILE   = "node_keys.json"


def _hash(payload: bytes) -> str:
    """Single point of hashing — swap algorithm here to migrate to PQ."""
    if HASH_ALGORITHM == "sha256":
        return hashlib.sha256(payload).hexdigest()
    if HASH_ALGORITHM == "sha3_256":
        return hashlib.sha3_256(payload).hexdigest()
    raise ValueError(f"Unsupported hash algorithm: {HASH_ALGORITHM}")



# 0b. POST-QUANTUM NODE IDENTITY (Dilithium2 signing key)

class PQNodeIdentity:
    """
    Manages a Dilithium2 key pair for this blockchain node.

    In a permissioned network (Hyperledger Fabric) each peer has a signing
    identity issued by the consortium CA. Here we generate and persist a
    local Dilithium2 key pair. Every block is signed with this key, and the
    signature is stored in the block's `pq_signature` field.

    This replaces ECDSA, which is broken by Shor's algorithm on a
    sufficiently large quantum computer.
    """
    def __init__(self, key_file: str = NODE_KEY_FILE):
        self.key_file = key_file
        self._dil = Dilithium2() if _PQ_AVAILABLE else None
        self.pk: Optional[bytes] = None
        self.sk: Optional[bytes] = None
        self._load_or_generate()

    def _load_or_generate(self):
        if not _PQ_AVAILABLE:
            return
        if os.path.exists(self.key_file):
            with open(self.key_file) as f:
                d = json.load(f)
            self.pk = bytes.fromhex(d["pk"])
            self.sk = bytes.fromhex(d["sk"])
        else:
            self.pk, self.sk = self._dil.keygen()
            with open(self.key_file, "w") as f:
                json.dump({"pk": self.pk.hex(), "sk": self.sk.hex(),
                           "scheme": "Dilithium2", "standard": "NIST FIPS 204"}, f, indent=2)
            print(f"[PQ] Generated Dilithium2 identity → {self.key_file}")

    def sign(self, payload: bytes) -> Optional[str]:
        """Sign payload, return hex-encoded Dilithium2 signature."""
        if not _PQ_AVAILABLE or self.sk is None:
            return None
        sig = self._dil.sign(self.sk, payload)
        return sig.hex()

    def verify(self, payload: bytes, sig_hex: str) -> bool:
        """Verify a Dilithium2 signature against this node's public key."""
        if not _PQ_AVAILABLE or self.pk is None or not sig_hex:
            return True  # unsigned mode: always pass
        try:
            sig = bytes.fromhex(sig_hex)
            return self._dil.verify(self.pk, payload, sig)
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return _PQ_AVAILABLE and self.pk is not None

    def pk_fingerprint(self) -> str:
        return self.pk.hex()[:32] + "..." if self.pk else "unavailable"


# 1. DATA STRUCTURES
# Event types (extended from v1):
#   GENESIS              — initial block
#   PROFILE_REGISTERED   — initial profile fingerprint
#   PROFILE_UPDATED      — profile retrained / migrated
#   PROFILE_VERIFIED     — integrity check result (active use of the chain)
#   INCIDENT             — anomaly detected by L/A/V cluster
#   TRANSACTION_BLOCKED  — auto-generated by smart contract on HIGH/CRITICAL
#   SECURITY_ALERT       — auto-generated by smart contract on HIGH/CRITICAL
#   MERKLE_ANCHOR        — periodic Merkle root for hybrid public verification
#   TAMPER_DETECTED      — explicit record that integrity check failed

EVENT_TYPES = {
    "GENESIS", "PROFILE_REGISTERED", "PROFILE_UPDATED", "PROFILE_VERIFIED",
    "INCIDENT", "TRANSACTION_BLOCKED", "SECURITY_ALERT",
    "MERKLE_ANCHOR", "TAMPER_DETECTED",
}


@dataclass
class BlockData:
    event_type: str
    employee_id: str
    profile_hash: Optional[str]      = None
    similarity_score: Optional[float] = None
    threshold: Optional[float]        = None
    text_snippet: Optional[str]       = None
    triggering_block: Optional[int]   = None   # for SC-generated blocks
    metadata: dict                    = field(default_factory=dict)


@dataclass
class Block:
    index: int
    timestamp: float
    data: BlockData
    previous_hash: str
    nonce: int = 0
    hash: str  = field(init=False)
    pq_signature: Optional[str] = None   # Dilithium2 signature over block hash

    def __post_init__(self):
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": asdict(self.data),
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return _hash(payload)

    def mine(self, difficulty: int = DIFFICULTY):
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self._compute_hash()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hash"] = self.hash
        d["pq_signature"] = self.pq_signature
        return d


# 2. SMART CONTRACT (Hyperledger chaincode simulation)
class SmartContract:
    """
    Mimics the deterministic on-chain logic of a Hyperledger Fabric chaincode.
    Whenever a new block is committed, all registered handlers are invoked
    against it. Handlers may emit additional blocks (auto-response).
    """
    def __init__(self):
        self._handlers: list[Callable[[Block, "CAPaFSBlockchain"], list[BlockData]]] = []

    def on_commit(self, handler):
        self._handlers.append(handler)
        return handler

    def fire(self, block: Block, chain: "CAPaFSBlockchain") -> list[BlockData]:
        emitted: list[BlockData] = []
        for h in self._handlers:
            emitted.extend(h(block, chain) or [])
        return emitted


# Built-in chaincode rules
def _rule_block_high_severity(block: Block, chain: "CAPaFSBlockchain") -> list[BlockData]:
    """If an INCIDENT is HIGH or CRITICAL → automatically block transaction."""
    if block.data.event_type != "INCIDENT":
        return []
    severity = block.data.metadata.get("severity")
    if severity not in ("HIGH", "CRITICAL"):
        return []
    return [
        BlockData(
            event_type="TRANSACTION_BLOCKED",
            employee_id=block.data.employee_id,
            triggering_block=block.index,
            metadata={
                "severity": severity,
                "auto_response": True,
                "rule": "block_on_high_severity",
            },
        ),
        BlockData(
            event_type="SECURITY_ALERT",
            employee_id=block.data.employee_id,
            triggering_block=block.index,
            metadata={
                "severity": severity,
                "channel": "SOC",
                "alert": f"Possible deepfake impersonation of {block.data.employee_id}",
            },
        ),
    ]



# 3. CORE BLOCKCHAIN
class CAPaFSBlockchain:

    def __init__(self,
                 employee_id: str = "CFO_001",
                 chain_file: str  = CHAIN_FILE,
                 anchor_file: str = ANCHOR_FILE,
                 difficulty: int  = DIFFICULTY,
                 anchor_interval: int = ANCHOR_INTERVAL,
                 load_existing: bool = True,
                 enable_smart_contract: bool = True):
        self.employee_id   = employee_id
        self.chain_file    = chain_file
        self.anchor_file   = anchor_file
        self.difficulty    = difficulty
        self.anchor_every  = anchor_interval
        self.chain: list[Block] = []
        self.smart_contract = SmartContract()

        # Post-quantum node identity (Dilithium2)
        if PQ_SIGNING and _PQ_AVAILABLE:
            self.pq_identity = PQNodeIdentity()
            print(f"[PQ] Dilithium2 identity loaded | pk: {self.pq_identity.pk_fingerprint()}")
        else:
            self.pq_identity = None
            if PQ_SIGNING:
                print("[PQ] pq_crypto.py not found — running without PQ signatures")

        if enable_smart_contract:
            self.smart_contract.on_commit(_rule_block_high_severity)

        if load_existing and os.path.exists(self.chain_file):
            self._load_chain()
            print(f"[BC] Loaded existing chain: {len(self.chain)} blocks "
                  f"(integrity: {'OK' if self.verify_chain(silent=True) else 'BROKEN'})")
        else:
            self._create_genesis()
            print(f"[BC] Created new chain (genesis)")

    #Block creation
    def _create_genesis(self):
        gen = Block(
            index=0,
            timestamp=time.time(),
            data=BlockData(
                event_type="GENESIS",
                employee_id=self.employee_id,
                metadata={
                    "system": "CAPaFS-DF",
                    "version": "2.0",
                    "hash_algorithm": HASH_ALGORITHM,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            ),
            previous_hash="0" * 64,
        )
        gen.mine(self.difficulty)
        self.chain.append(gen)
        self._save_chain()

    def _add_block(self, data: BlockData, fire_chaincode: bool = True) -> Block:
        prev = self.chain[-1]
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=prev.hash,
        )
        block.mine(self.difficulty)

        # Post-quantum block signing with Dilithium2
        if self.pq_identity and self.pq_identity.available:
            block.pq_signature = self.pq_identity.sign(block.hash.encode())

        self.chain.append(block)

        # Smart-contract auto-response (chaincode)
        if fire_chaincode:
            for emitted in self.smart_contract.fire(block, self):
                self._add_block(emitted, fire_chaincode=False)  # avoid recursion

        # Hybrid anchoring
        if len(self.chain) % self.anchor_every == 0:
            self._publish_anchor()

        self._save_chain()
        return block

    # Public API: profile lifecycle
    def register_profile(self, profile_bytes: bytes,
                         model_params: Optional[dict] = None) -> Block:
        ph = _hash(profile_bytes)
        return self._add_block(BlockData(
            event_type="PROFILE_REGISTERED",
            employee_id=self.employee_id,
            profile_hash=ph,
            metadata=model_params or {},
        ))

    def update_profile(self, profile_bytes: bytes, reason: str = "") -> Block:
        ph = _hash(profile_bytes)
        return self._add_block(BlockData(
            event_type="PROFILE_UPDATED",
            employee_id=self.employee_id,
            profile_hash=ph,
            metadata={"reason": reason},
        ))

    # Public API: ACTIVE profile-integrity verification (key improvement)
    def verify_profile_against_chain(self, profile_bytes: bytes) -> tuple[bool, str]:
        """
        Compare hash of the locally-loaded profile against the latest
        PROFILE_REGISTERED / PROFILE_UPDATED block. Returns (ok, reason).

        Pipelines MUST call this before classification. If ok == False, the
        local model has been tampered with and must not be used.
        """
        expected = self.get_current_profile_hash()
        if expected is None:
            return False, "no profile registered in chain"
        actual = _hash(profile_bytes)
        if actual == expected:
            self._add_block(BlockData(
                event_type="PROFILE_VERIFIED",
                employee_id=self.employee_id,
                profile_hash=actual,
                metadata={"result": "OK"},
            ))
            return True, "match"
        # Mismatch  record TAMPER_DETECTED
        self._add_block(BlockData(
            event_type="TAMPER_DETECTED",
            employee_id=self.employee_id,
            profile_hash=actual,
            metadata={
                "expected": expected,
                "actual": actual,
                "severity": "CRITICAL",
                "alert": "Local profile does not match registered hash. Refuse classification.",
            },
        ))
        return False, f"profile hash mismatch (expected={expected[:16]}…, actual={actual[:16]}…)"

    def get_current_profile_hash(self) -> Optional[str]:
        for b in reversed(self.chain):
            if b.data.event_type in ("PROFILE_REGISTERED", "PROFILE_UPDATED"):
                return b.data.profile_hash
        return None

    #Public API: incidents
    def log_incident(self, text: str, similarity: float, threshold: float,
                     profile_hash: Optional[str] = None) -> Block:
        snippet = (text[:77] + "...") if len(text) > 80 else text
        sev = self._severity(similarity, threshold)
        n_inc = sum(1 for b in self.chain if b.data.event_type == "INCIDENT") + 1
        return self._add_block(BlockData(
            event_type="INCIDENT",
            employee_id=self.employee_id,
            profile_hash=profile_hash or self.get_current_profile_hash(),
            similarity_score=round(similarity, 6),
            threshold=round(threshold, 6),
            text_snippet=snippet,
            metadata={
                "incident_number": n_inc,
                "deviation": round(threshold - similarity, 6),
                "severity": sev,
            },
        ))

    @staticmethod
    def _severity(sim: float, threshold: float) -> str:
        d = threshold - sim
        if d > 0.30: return "CRITICAL"
        if d > 0.15: return "HIGH"
        if d > 0.05: return "MEDIUM"
        return "LOW"

    #Integrity verification
    def verify_chain(self, silent: bool = False) -> bool:
        for i in range(1, len(self.chain)):
            curr, prev = self.chain[i], self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                if not silent:
                    print(f"[BC] ✗ Broken link at block #{i}")
                return False
            if curr.hash != curr._compute_hash():
                if not silent:
                    print(f"[BC] ✗ Hash mismatch at block #{i} "
                          f"(stored {curr.hash[:12]}…, recomputed {curr._compute_hash()[:12]}…)")
                return False
        return True

    def detect_tampering(self) -> list[dict]:
        """
        Returns a list of detected tampering events with the offending block index
        and a human-readable reason. Used by the tamper demo.
        """
        problems = []
        for i in range(1, len(self.chain)):
            curr, prev = self.chain[i], self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                problems.append({"block": i, "reason": "broken previous_hash link"})
            if curr.hash != curr._compute_hash():
                problems.append({"block": i, "reason": "stored hash differs from recomputed hash"})
        return problems

    # Merkle anchoring (hybrid architecture simulation) 
    def _merkle_root(self, hashes: list[str]) -> str:
        if not hashes:
            return "0" * 64
        layer = list(hashes)
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            layer = [_hash((layer[i] + layer[i + 1]).encode())
                     for i in range(0, len(layer), 2)]
        return layer[0]

    def _publish_anchor(self):
        """Compute Merkle root of all blocks and write to external anchor file
        (simulates publication to a public chain like Ethereum)."""
        root = self._merkle_root([b.hash for b in self.chain])
        anchor = {
            "anchored_at_block": len(self.chain) - 1,
            "timestamp": time.time(),
            "merkle_root": root,
            "chain_length": len(self.chain),
            "hash_algorithm": HASH_ALGORITHM,
        }
        anchors = []
        if os.path.exists(self.anchor_file):
            with open(self.anchor_file, "r") as f:
                anchors = json.load(f)
        anchors.append(anchor)
        with open(self.anchor_file, "w") as f:
            json.dump(anchors, f, indent=2)
        # Record the anchor on-chain as well (without re-firing chaincode)
        self.chain.append(Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=BlockData(
                event_type="MERKLE_ANCHOR",
                employee_id=self.employee_id,
                metadata=anchor,
            ),
            previous_hash=self.chain[-1].hash,
        ))
        self.chain[-1].mine(self.difficulty)

    # Persistence
    def _save_chain(self):
        with open(self.chain_file, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in self.chain], f, ensure_ascii=False, indent=2)

    def _load_chain(self):
        with open(self.chain_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chain = []
        for item in raw:
            bd = BlockData(**item["data"])
            block = Block.__new__(Block)
            block.index = item["index"]
            block.timestamp = item["timestamp"]
            block.data = bd
            block.previous_hash = item["previous_hash"]
            block.nonce = item["nonce"]
            block.hash = item["hash"]
            block.pq_signature = item.get("pq_signature")
            self.chain.append(block)

    def verify_pq_signatures(self) -> dict:
        """
        Verify all Dilithium2 block signatures.

        Critical: we verify the signature against the RECOMPUTED hash
        (from block data), NOT the stored hash. This catches the attack
        where an adversary modifies block data but leaves the old hash
        and old signature in place — the recomputed hash will differ from
        what was signed, so the signature check fails.

        This gives TWO independent detection layers for data tampering:
          1. SHA3 hash-chain integrity (stored hash vs recomputed hash)
          2. Dilithium2 signature (signature vs recomputed hash)
        """
        if not (self.pq_identity and self.pq_identity.available):
            return {"available": False, "reason": "pq_crypto not loaded"}

        signed   = [b for b in self.chain if b.pq_signature]
        invalid  = []
        for b in signed:
            # Recompute the hash from scratch — do not trust the stored value
            recomputed = b._compute_hash()
            if not self.pq_identity.verify(recomputed.encode(), b.pq_signature):
                invalid.append(b.index)

        return {
            "available":     True,
            "total_blocks":  len(self.chain),
            "signed_blocks": len(signed),
            "valid_sigs":    len(signed) - len(invalid),
            "invalid_sigs":  len(invalid),
            "tampered_blocks": invalid,
            "pq_scheme":     "Dilithium2 (NIST FIPS 204)",
            "all_valid":     len(invalid) == 0,
        }

    # Reporting
    def summary(self) -> dict:
        from collections import Counter
        types = Counter(b.data.event_type for b in self.chain)
        incidents = [b for b in self.chain if b.data.event_type == "INCIDENT"]
        sevs = Counter(b.data.metadata.get("severity") for b in incidents)
        size = os.path.getsize(self.chain_file) if os.path.exists(self.chain_file) else 0
        signed = sum(1 for b in self.chain if b.pq_signature)
        return {
            "blocks": len(self.chain),
            "events": dict(types),
            "incidents_by_severity": dict(sevs),
            "integrity": self.verify_chain(silent=True),
            "hash_algorithm": HASH_ALGORITHM,
            "pq_signing": f"Dilithium2 ({signed}/{len(self.chain)} blocks signed)" if signed else "disabled",
            "size_bytes": size,
            "size_per_block_bytes": round(size / max(len(self.chain), 1), 1),
            "current_profile_hash": (self.get_current_profile_hash() or "")[:16] + "...",
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "═" * 70)
        print("  BLOCKCHAIN SUMMARY")
        print("═" * 70)
        for k, v in s.items():
            print(f"  {k:<26}: {v}")
        print("═" * 70)



# 4. INDEPENDENT VERIFIER (third-party / regulator)
class IndependentVerifier:
    """
    Verifies a chain WITHOUT access to the bank's internal model.
    Demonstrates the regulator/auditor use case described in the thesis.
    """
    def __init__(self, chain_file: str = CHAIN_FILE, anchor_file: str = ANCHOR_FILE):
        self.chain_file  = chain_file
        self.anchor_file = anchor_file

    def verify(self) -> dict:
        with open(self.chain_file) as f:
            raw = json.load(f)
        # 1. Hash-link integrity
        for i in range(1, len(raw)):
            recomputed = _hash(json.dumps({
                "index": raw[i]["index"],
                "timestamp": raw[i]["timestamp"],
                "data": raw[i]["data"],
                "previous_hash": raw[i]["previous_hash"],
                "nonce": raw[i]["nonce"],
            }, sort_keys=True, ensure_ascii=False).encode())
            if recomputed != raw[i]["hash"]:
                return {"ok": False, "reason": f"block #{i} hash recomputation failed"}
            if raw[i]["previous_hash"] != raw[i - 1]["hash"]:
                return {"ok": False, "reason": f"block #{i} broken link to previous"}

        # 2. Anchor consistency (hybrid mode)
        anchor_ok = True
        if os.path.exists(self.anchor_file):
            with open(self.anchor_file) as f:
                anchors = json.load(f)
            if anchors:
                latest = anchors[-1]
                # Recompute Merkle root over hashes up to the anchored block
                hashes = [b["hash"] for b in raw[: latest["anchored_at_block"] + 1]]
                if hashes:
                    layer = list(hashes)
                    while len(layer) > 1:
                        if len(layer) % 2 == 1:
                            layer.append(layer[-1])
                        layer = [_hash((layer[i] + layer[i + 1]).encode())
                                 for i in range(0, len(layer), 2)]
                    if layer[0] != latest["merkle_root"]:
                        anchor_ok = False

        return {
            "ok": True,
            "blocks": len(raw),
            "anchor_consistent": anchor_ok,
            "verifier": "third-party (no internal model required)",
        }


# 5. SELF-EVALUATION (for "evaluate your solution" thesis requirement)
def benchmark(n_incidents: int = 100,
              employee_id: str = "BENCH",
              chain_file: str = "bench_chain.json") -> dict:
    """
    Measures: throughput (blocks/s), per-block storage cost, verification cost.
    Used to populate a quantitative table in the thesis.
    """
    if os.path.exists(chain_file):
        os.remove(chain_file)
    bc = CAPaFSBlockchain(employee_id=employee_id, chain_file=chain_file,
                          load_existing=False, enable_smart_contract=False)
    bc.register_profile(b"benchmark_profile", {"n_train": 0})

    t0 = time.perf_counter()
    for i in range(n_incidents):
        bc.log_incident(f"benchmark message #{i}", 0.10, 0.40)
    t_write = time.perf_counter() - t0

    t1 = time.perf_counter()
    ok = bc.verify_chain(silent=True)
    t_verify = time.perf_counter() - t1

    size = os.path.getsize(chain_file)
    return {
        "n_blocks": len(bc.chain),
        "n_incidents_requested": n_incidents,
        "write_time_s": round(t_write, 4),
        "blocks_per_second": round(len(bc.chain) / t_write, 1) if t_write > 0 else 0,
        "verify_time_s": round(t_verify, 6),
        "verify_blocks_per_second": round(len(bc.chain) / t_verify, 0) if t_verify > 0 else 0,
        "chain_size_bytes": size,
        "bytes_per_block": round(size / len(bc.chain), 1),
        "integrity": ok,
        "hash_algorithm": HASH_ALGORITHM,
        "difficulty": bc.difficulty,
    }


# 6. DEMO / ENTRY POINT
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print(json.dumps(benchmark(n_incidents=200), indent=2))
        sys.exit(0)

    # Fresh demo chain
    if os.path.exists(CHAIN_FILE):  os.remove(CHAIN_FILE)
    if os.path.exists(ANCHOR_FILE): os.remove(ANCHOR_FILE)

    bc = CAPaFSBlockchain(employee_id="CFO_DEMO", load_existing=False)
    bc.register_profile(b"demo_profile_v1", {"model": "TF-IDF", "n_features": 1500})

    # Mix of severities to trigger smart contract on HIGH/CRITICAL
    bc.log_incident("Wire $2M offshore now, no approval", 0.05, 0.45)  # CRITICAL
    bc.log_incident("Transfer urgent — $1.5M today",       0.18, 0.45)  # HIGH
    bc.log_incident("Slight stylistic deviation",          0.42, 0.45)  # LOW

    # Active integrity verification — should pass
    ok, why = bc.verify_profile_against_chain(b"demo_profile_v1")
    print(f"\n[BC] Profile verification (legitimate): ok={ok} ({why})")

    # Active integrity verification — should fail (tampered profile)
    ok, why = bc.verify_profile_against_chain(b"demo_profile_v1_TAMPERED")
    print(f"[BC] Profile verification (tampered):   ok={ok} ({why})")

    bc.print_summary()

    # Independent verification by regulator
    print("\n--- INDEPENDENT VERIFIER (regulator view) ---")
    print(json.dumps(IndependentVerifier().verify(), indent=2))