import numpy as np

from scripts.derive_aeb_request_timing import _stats


def test_stats_uses_standard_linear_quantiles():
    values = [1.0, 2.0, 3.0, 10.0]
    result = _stats(values)
    assert result['p5'] == round(float(np.quantile(values, 0.05)), 3)
    assert result['median'] == round(float(np.median(values)), 3)
    assert result['p95'] == round(float(np.quantile(values, 0.95)), 3)
