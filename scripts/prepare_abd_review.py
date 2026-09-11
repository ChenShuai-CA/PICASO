"""Create a bounded review queue; never infer AEB/collision labels from flags."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scenario_lab.data import inspect_abd


def main():
    inventory=ROOT/'research_audit_20260910/expanded_inventory/all_runs_inventory.csv'
    output=ROOT/'runs/20260911_abd_review'
    output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(inventory.open(encoding='utf-8-sig')))
    chosen={}
    for row in rows:
        if row['data_role']=='protocol' and row['scenario_acronym'] in ('CCRs','CPTA','CCFT'):
            chosen.setdefault((row['brand'],row['scenario_acronym']),row)
    details=[]
    review=[]
    for row in list(chosen.values())[:24]:
        path=ROOT/'Data/ABD_Data'/row['rel_path']
        info=inspect_abd(path,max_rows=300)
        channels=[c for c in info['channels'] if any(k in c['name'].lower() for k in ('brake','aeb','fcw','abort','speed','time'))]
        detail=dict(run=row['rel_path'],vehicle_folder=row['brand'],scenario=row['scenario_acronym'],
                    config=info['config'],time=info['time'],parse_status=info['parse_status'],
                    selected_channels=channels,reasons=info['reasons'],
                    suitable_for_aeb_calibration=False,collision_label=None)
        details.append(detail)
        review.append(dict(run=row['rel_path'],vehicle_folder=row['brand'],scenario=row['scenario_acronym'],
                           use_brake_robot=info['config'].get('use_brake_robot'),parse_status=info['parse_status'],
                           confirmed_vehicle_configuration='',brake_event_source='',aeb_channel_and_unit='',
                           time_alignment_verified='',event_window='',calibration_decision='',reviewer=''))
    with (output/'manual_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(review[0]) if review else ['run'])
        writer.writeheader();writer.writerows(review)
    (output/'evidence.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(inspected_runs=len(details),sample_rows_per_run=300,automatic_aeb_labels_created=0,
                limitation='Header/sample-window inspection only; human control attribution and full event windows required',
                scenarios=sorted({r['scenario'] for r in review}))
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
