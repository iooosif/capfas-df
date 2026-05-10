from blockchain_v2 import CAPaFSBlockchain

bc = CAPaFSBlockchain(load_existing=True)
report = bc.verify_pq_signatures()

print(f"Scheme       : {report['pq_scheme']}")
print(f"Total blocks : {report['total_blocks']}")
print(f"Signed       : {report['signed_blocks']}")
print(f"Valid sigs   : {report['valid_sigs']}")
print(f"Invalid sigs : {report['invalid_sigs']}")
print(f"All valid    : {report['all_valid']}")