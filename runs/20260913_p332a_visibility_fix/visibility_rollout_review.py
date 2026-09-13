import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('fixture', ROOT / 'tests/test_p33_model.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
folder = ROOT / 'runs/20260913_p332a_visibility_fix'
manifest = json.loads((folder / 'ARTIFACT_MANIFEST.json').read_text())
bad = []
for name, entry in manifest['artifacts'].items():
    with (folder / name).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != entry['sha256']:
        bad.append(name)
print('hash_mismatches', bad, flush=True)
from scenario_lab.p33_model import ARSceneV1
import numpy as np
with np.load(ROOT / 'runs/20260913_p331_data_pipeline/smoke/motion_codebook_v1.npz') as data:
    codebook = data['centroids'].copy()
model = ARSceneV1(t.CONFIG, codebook).eval()
checkpoint = torch.load(folder / 'm2_final_checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'])
batch = t._make_batch(batch_size=1)
batch['pairwise_visibility_mask'][:,0,:-1,6] = False
changed = copy.deepcopy(batch)
changed['agent_history'][:,6,:-1,:4] += 1000
a = t.to_torch_batch(batch)
b = t.to_torch_batch(changed)
with torch.no_grad():
    fa = model(a, a['motion_token_target'])['motion_token_logits'][:,0]
    fb = model(b, b['motion_token_target'])['motion_token_logits'][:,0]
print('fixed_teacher_equal', torch.equal(fa, fb), flush=True)
ra = model.rollout(a, num_samples=6, seed=123)
rb = model.rollout(b, num_samples=6, seed=123)
print('rollout_query0_token_differences', int((ra['tokens'][:,:,0] != rb['tokens'][:,:,0]).sum()), flush=True)
print('rollout_all_token_differences', int((ra['tokens'] != rb['tokens']).sum()), flush=True)
print('rollout_query0_max_xy_difference', float((ra['trajectories'][:,:,0]-rb['trajectories'][:,:,0]).abs().max()), flush=True)
