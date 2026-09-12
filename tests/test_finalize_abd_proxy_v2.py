from pathlib import Path

from scripts.finalize_abd_proxy_v2 import (
    classify_aeb_review,
    classify_fcw_review,
    normalize_run,
    read_csv_auto,
)


def test_review_csv_reader_accepts_excel_gb18030(tmp_path):
    path = tmp_path / "review.csv"
    path.write_bytes("run,driver_intervention\na.txt,人工接管\n".encode("gb18030"))
    assert read_csv_auto(path)[0]["driver_intervention"] == "人工接管"


def test_no_takeover_is_eligible_for_full_aeb_window():
    result = classify_aeb_review("none_confirmed")
    assert result["aeb_response_eligibility"] == "eligible_full_braking_window"


def test_manual_after_completed_aeb_stop_is_censored_not_discarded():
    result = classify_aeb_review(
        "AEB触发了，刹停避撞之后又松开了刹车导致溜车，最后人工接管了"
    )
    assert result["intervention_class"] == "manual_after_aeb_stop"
    assert result["aeb_response_eligibility"] == "eligible_through_first_stop_only"


def test_fcw_only_manual_takeover_is_never_aeb_evidence():
    value = "FCW场景只测FCW，不测AEB，这里是听到报警声后就人工接管了"
    assert classify_fcw_review(value).startswith("manual_after_fcw_audio")


def test_normalize_run_removes_workspace_data_prefix():
    assert normalize_run("Data/ABD_Data/car/run.txt") == "car/run.txt"
