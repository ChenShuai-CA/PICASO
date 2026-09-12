from pathlib import Path

import numpy as np

from scripts.audit_abd_fcw_audio import (
    fcw_intervention_class,
    find_can_user_mappings,
    first_rising_time,
    read_review_csv,
    read_spec,
)


def test_spec_mapping_identifies_sound_alarm(tmp_path: Path):
    path = tmp_path / 'run.spec'
    path.write_text(
        'Description=CCRs FCW\n'
        'TimeTolerance3Channel1=CAN User Defined 1 (SoundAlarm1)\n'
        'TimeTolerance3MinTrigger1=0.5\n'
        'TimeTolerance3MaxTrigger1=1.5\n'
        'TimeTolerance3TriggerTime=0\n'
        'TimeTolerance3TriggerDelayTime=1\n',
        encoding='latin-1')
    mappings = find_can_user_mappings(read_spec(path), str(path))
    assert len(mappings) == 1
    assert mappings[0]['ttt_index'] == 3
    assert mappings[0]['can_user_defined_index'] == 1
    assert mappings[0]['semantic_status'] == 'avad3_fcw_audio_explicit_sound_label'


def test_fcw_context_supports_unlabelled_operator_convention():
    spec = {'Description': 'CCRs FCW',
            'TimeTolerance1Channel1': 'CAN User Defined 2'}
    mapping = find_can_user_mappings(spec)[0]
    assert mapping['semantic_status'] == (
        'avad3_fcw_audio_by_operator_convention_in_fcw_context')


def test_first_rising_time_uses_first_zero_to_one_edge():
    time = np.asarray([0.0, 0.01, 0.02, 0.03])
    values = np.asarray([0.0, 0.0, 1.0, 0.0])
    assert first_rising_time(time, values) == 0.02


def test_review_reader_accepts_excel_gb18030_and_classifies_fcw_takeover(tmp_path):
    path = tmp_path / 'review.csv'
    note = 'FCW场景只测FCW，不测AEB，这里是听到报警声后就人工接管了'
    path.write_bytes(f'run,driver_intervention\na.txt,{note}\n'.encode('gb18030'))
    value = read_review_csv(path)[0]['driver_intervention']
    assert value == note
    assert fcw_intervention_class(value).startswith('manual_after_fcw_audio')
