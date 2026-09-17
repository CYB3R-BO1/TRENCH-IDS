import json
with open('runs/meta_transfer/B0_poc/meta_episodes/episodes_train_manifest.json') as f:
    manifest = json.load(f)
print('Type:', type(manifest))
print('Length:', len(manifest))
if len(manifest) > 0:
    print('First episode keys:', list(manifest[0].keys()) if isinstance(manifest[0], dict) else 'N/A')