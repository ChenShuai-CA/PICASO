"""Read-only evidence probes for the independent P3.3.4 review; dev only."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/20260915_p334_kinematics'
OUT = RUN / 'codex_review'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)

def read(name):
    return json.loads((RUN / name).read_text())

b = read('PHASE_B_METRICS.json')
c = read('PHASE_C_METRICS.json')
orig = b['variants']['orig']
macro = lambda x: np.mean([v['min_ade'] for v in x['group_level'].values()])
result = {'orig_macro': macro(orig), 'c_macro': macro(c),
          'macro_relative_pct': (macro(c)/macro(orig)-1)*100,
          'macro_absolute_margin_m': macro(orig)*1.05-macro(c),
          'sampling_equal': b['sampling'] == c['sampling'],
          'checkpoint_hashes': {}, 'error_deltas_pct': {}}
for record in (b,c):
    path = ROOT / record['checkpoint']['path']
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    result['checkpoint_hashes'][record['phase']] = {
        'sha256': digest, 'matches': digest == record['checkpoint']['sha256']}
for source in ('waymo','interaction'):
    result['error_deltas_pct'][source] = {
        key: (c['group_level'][source][key]/orig['group_level'][source][key]-1)*100
        for key in ('min_ade','min_fde','min_ade_joint','min_fde_joint')}
log = [json.loads(s) for s in (RUN/'phase_c_train/ARCH_TRAIN_LOG.jsonl').read_text().splitlines()]
result['training'] = {'epochs': [r['epoch'] for r in log],
    'best': min(r['source_macro_minade_at_6'] for r in log),
    'last_update': log[-1]['update_index']}

# A saturated but legal head is a deterministic counterexample to start cap.
spec = importlib.util.spec_from_file_location('fixture', ROOT/'tests/test_p334_phase_c_kinematic.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
model, batch = fixture._model(), fixture._tiny_batch()
model.eval()
with torch.no_grad():
    model.init_head.weight.zero_()
    model.init_head.bias.copy_(torch.tensor([100.,0.]))
    model.jerk_head.weight.zero_()
    model.jerk_head.bias.zero_()
    kin = model(batch, teacher_tokens=batch['motion_token_target'])['kinematic']
    v0, _ = model._history_tail_velocity(batch)
    implied = torch.linalg.vector_norm((kin['v_seq'][:,:,1]-v0)/model.h,dim=-1)
result['start_cap_counterexample'] = {'claimed_cap': 9.5,
    'observed_implied_accel': implied.tolist(),
    'vehicle_a1_a2': kin['a_seq'][0,0,:2].tolist()}

# Fixed prefix, changed final token: the last token cannot influence any output.
model = fixture._model(seed=19).eval()
tokens = batch['motion_token_target'].clone()
changed = tokens.clone()
changed[:,:,-1] = (changed[:,:,-1]+1) % 128
with torch.no_grad():
    first = model.trajectory_from_tokens(tokens,batch)
    second = model.trajectory_from_tokens(changed,batch)
result['final_token_intervention_max_position_delta'] = float((first-second).abs().max())
sys.path.insert(0, str(ROOT/'research_tasks'))
from p334_phase_b import candidate_diversity
toy = np.zeros((2,1,50,2))
toy[1,:,:,0] = 1.0
result['diversity_one_meter_apart'] = {
    'reported': candidate_diversity(toy,np.ones((1,50),dtype=bool)),
    'expected_time_mean': 1.0}
result['gt_continuity'] = {k:v['continuity'] for k,v in read('attribution_gt.json')['table'].items()}

# Scan only dev history, retaining identifiers for implausible initial speeds.
from scenario_lab.p33_dataset import ShardCatalog, iter_split_batches
from scenario_lab.p33_metrics import evaluate_agent_mask
from scenario_lab.p33_spec import load_p33_config
catalog = ShardCatalog.from_manifest(ROOT/'runs/20260913_p333_architecture/data/DATASET_MANIFEST.json', load_p33_config())
outliers = []
samples = 0
for raw in iter_split_batches(catalog, split='dev', batch_size=16):
    samples += len(raw['sample_id'])
    v = np.diff(raw['agent_history'][:,:,-2:,:2].astype(np.float64),axis=2)[:,:,0]*10
    speed = np.linalg.norm(v,axis=-1)
    usable = raw['state_valid_mask'][:,:,-2:].all(axis=-1)
    for i in range(len(raw['sample_id'])):
        mask = evaluate_agent_mask(raw['agent_present_mask'][i],raw['agent_role'][i],raw['future_valid_mask'][i])
        for a in np.flatnonzero(mask & usable[i] & (speed[i]>35)):
            outliers.append({'sample_id': str(raw['sample_id'][i]),
                'source': str(raw['sample_source'][i]), 'agent_slot': int(a),
                'history_speed_mps': float(speed[i,a]),
                'history_xy': raw['agent_history'][i,a,-2:,:2].tolist()})
result['dev_samples_scanned'] = samples
result['history_speed_outliers'] = outliers
(OUT/'EVIDENCE.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
