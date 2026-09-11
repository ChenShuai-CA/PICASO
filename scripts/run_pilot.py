"""Reproducible Linux GPU pilot; outputs are engineering checks, not paper claims."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--updates',type=int,default=12)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--count',type=int,default=20)
    args=parser.parse_args()
    assert platform.system()=='Linux', 'Use WSL2 Ubuntu'
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    corpus=Path(args.corpus).resolve()
    corpus_report=json.loads(corpus.with_name('report.json').read_text())
    if not corpus_report.get('ready_for_joint_pilot'):
        raise ValueError('Corpus lacks both sources/roles or train/validation coverage')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    def run(name,command):
        job={'name':name,'command':[sys.executable,'-m','scenario_lab',*map(str,command)],'status':'running'}
        jobs.append(job)
        (output/'jobs.json').write_text(json.dumps(jobs,indent=2))
        print('START '+name,flush=True)
        start=time.perf_counter()
        with (output/(name+'.log')).open('w') as log:
            completed=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT)
        job.update(status='passed' if completed.returncode==0 else 'failed',returncode=completed.returncode,elapsed_s=time.perf_counter()-start)
        (output/'jobs.json').write_text(json.dumps(jobs,indent=2))
        print(json.dumps({k:v for k,v in job.items() if k!='command'}),flush=True)
        if completed.returncode:
            raise RuntimeError('Pilot stopped; inspect '+str(output/(name+'.log')))
    run('pretrain',['pretrain','--corpus',corpus,'--output',output/'prior','--epochs',args.epochs,'--device','cuda'])
    prior=output/'prior/prior.pt'
    models=[]
    variants=[('mixed_seed7','mixed','mappo',7,[]),('mixed_seed17','mixed','mappo',17,[]),
              ('mixed_seed27','mixed','mappo',27,[]),('single_seed7','single','ppo',7,[]),
              ('dual_seed7','dual','mappo',7,[]),('mixed_no_prior_seed7','mixed','mappo',7,['no_prior']),
              ('mixed_no_robust_seed7','mixed','mappo',7,['--no-robust'])]
    for name,mode,algorithm,seed,flags in variants:
        command=['train','--output',output/name,'--updates',args.updates,'--episodes-per-update',4,
                 '--mode',mode,'--algorithm',algorithm,'--seed',seed,'--device','cuda']
        if 'no_prior' not in flags:command+=['--pretrained',prior]
        command += [flag for flag in flags if flag!='no_prior']
        run(name,command)
        models.append((name,output/name/'policy.pt'))
    run('script_eval',['evaluate','--output',output/'eval_script','--count',args.count,'--seed',1000])
    for name,policy in [('prior',prior),*models]:
        run('eval_'+name,['evaluate','--policy',policy,'--output',output/('eval_'+name),
                         '--count',args.count,'--seed',1000,'--device','cuda'])
    main_policy=output/'mixed_seed7/policy.pt'
    run('held_controller',['evaluate','--policy',main_policy,'--output',output/'held_controller',
                          '--count',args.count,'--seed',1000,'--controller','ttc','--device','cuda'])
    run('perturbations',['evaluate','--policy',main_policy,'--output',output/'perturbations',
                        '--count',10,'--seed',2000,'--perturbations',10,'--device','cuda'])
    for branch in ('single','dual'):
        for kind in ('parameters','trajectory'):
            run('search_'+branch+'_'+kind,['search','--branch',branch,'--kind',kind,'--budget',40,
                                         '--output',output/('search_'+branch+'_'+kind)])
    # Benchmark alone after training/evaluation; avoid concurrent GPU jobs from this runner.
    run('latency',['benchmark','--policy',main_policy,'--steps',10000,
                   '--output',output/'latency.json','--device','cuda'])
    run('replay',['replay',output/'eval_mixed_seed7/dual_0000_00.json'])
    (output/'completed.json').write_text(json.dumps(dict(status='completed',jobs=len(jobs),
        results_kind='small_budget_engineering_pilot_not_publication_evidence',
        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':
    main()
