"""
make_suspicious.py — generates test_suspicious.csv
Messages are intentionally designed to NOT resemble CFO business correspondence:
- Short, blunt, imperative
- Atypical vocabulary: wire, urgent, secret, bypass, offshore
- No business-like phrases: no "please find attached," no "as per"
- Numbers and invoices are prominent
- Time pressure, secrecy, circumvention of procedures
"""
import pandas as pd, random

SUSPICIOUS = [
    # BEC immediate transfer requests
    "Wire $2.4M to account DE89370400440532013000 right now. Do not wait for approval.",
    "Send $850k to Cayman Private Bank account immediately. Confidential — tell no one.",
    "Transfer $3M offshore today. Bypass the finance team. I'll explain later.",
    "Move funds $1.2M to account CH9300762011623852957 before 3pm. No paperwork.",
    "URGENT: wire $475000 to GB82WEST12345698765432 now. CEO order. No questions.",
    "Get $6M out to Nexus Capital today. Do not log this. Personal request from me.",
    "Send all liquid funds to Baltic Financial Group account immediately. Secret.",
    "I need $500k wired to my personal account today. Do not tell the CFO office.",
    "Transfer $2M to offshore account now. Skip dual approval. I authorize this.",
    "Wire $150k to Vertex Solutions now. Do not create purchase order or ticket.",

    # Change of account details
    "Change all payments for Alpine Trading to new account XX73819 at First National Offshore.",
    "Update vendor bank details for GlobeServ Inc to CH9300762011623852957. Do it now.",
    "New wire instructions for all Meridian invoices: account DE89370400440532013000. Update immediately.",
    "Reroute Nexus Capital payments to new offshore account. Do not contact vendor to verify.",
    "Switch all ClearPath Finance payments to account GB82WEST12345698765432 today.",

    # Request for credentials
    "Send me banking portal login and password right now. Personal email only.",
    "Give the treasury system credentials to Robert Chen. Do not go through IT.",
    "I need wire authorization token for main account. Send to my Gmail directly.",
    "Share login details for payment system with external advisor Diana Walsh immediately.",
    "Forward banking passwords to my personal phone. Do not use corporate email.",

    # Time pressure
    "DO IT NOW: $3M wire to account CH9300762011623852957. We lose the deal in 1 hour.",
    "STOP everything. Wire $850k to GB82WEST12345698765432 before market close. NOW.",
    "Emergency: move $2M out immediately or we lose the contract. No time to explain.",
    "30 minutes: send $475000 to offshore account or deal collapses. Just do it.",
    "CRITICAL — wire $6.5M right now. Account XX91827. Do not delay.",

    # Secrecy
    "Do not tell anyone about this transfer. Send $2M to account DE89370400440532013000.",
    "Keep this between us: move $1M to Cayman account today. No paper trail.",
    "Secret payment — $3M to account CH9300762011623852957. Do not mention in reports.",
    "This is confidential. Wire $500k offshore. Delete this message after.",
    "Do not document. Send $850k to account GB82WEST12345698765432. Personal order.",

    # Circumvention of procedures
    "Skip normal approval. Send $2M to Alpine Trading now. I take responsibility.",
    "Ignore the dual-sign rule today. Wire $3M immediately. My authorization.",
    "Process payment without committee sign-off. $1.2M to Nexus Capital offshore.",
    "Bypass finance controls for this one. $475000 to First National Offshore. Now.",
    "No need for PO or invoice. Send $6M to Vertex Solutions account today.",

    # Atypical structure / fraud
    "You are selected for urgent fund release. Process $2M to account XX91827 today.",
    "Regulatory issue — must move $3M before auditors arrive tomorrow morning.",
    "CEO asked me personally to wire $850k offshore. Do not involve compliance.",
    "Investment opportunity — send $500k to account CH9300762011623852957 immediately.",
    "Fake audit risk — move funds $2.4M to new account DE89370400440532013000 tonight.",
]

random.Random(42).shuffle(SUSPICIOUS)
pd.DataFrame({"text": SUSPICIOUS}).to_csv("test_suspicious.csv", index=False)
print(f"[✓] Generated {len(SUSPICIOUS)} suspicious messages → test_suspicious.csv")
for msg in SUSPICIOUS[:3]:
    print(f"  ► {msg}")