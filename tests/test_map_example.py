"""The MAP of the shipped example.

The optimizer used to move the transit to absorb the unfitted systematics and
converge on a deep grazing transit: ror 0.131, b 0.85, t0 1.079.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.slow

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'hip67522c'


def test_example_map_keeps_the_transit_at_the_observed_ingress(tmp_path):
    from timer import fit

    wd = tmp_path / 'hip67522c'
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))
    with open(wd / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(wd / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)
    fit_params['clobber'] = True
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    ror = float(np.ravel(tf.map_soln['ror'])[0])
    t0 = float(np.ravel(tf.map_soln['t0'])[0])
    assert 0.05 < ror < 0.08
    assert t0 == pytest.approx(1.031, abs=0.01)

    # The staged custom optimizer reaches -879.1. If it fails, build falls
    # back to pm.find_MAP, which lands on a similar transit at about -927, so
    # the parameter checks above would not notice.
    value_names = [var.name for var in tf.model.value_vars]
    point = {k: tf.map_soln[k] for k in value_names}
    logp = float(sum(tf.model.point_logps(point).values()))
    assert logp > -900
