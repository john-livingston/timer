import numpy as np
import pymc as pm
import pytest

from timer import optim


@pytest.fixture
def tiny_model():
    with pm.Model() as model:
        pm.Normal('a', 0.0, 1.0)
        pm.Normal('y_observed', mu=0.0, sigma=1.0, observed=np.zeros(3))
    return model


def test_ctrl_c_aborts_instead_of_returning_the_unoptimized_init_point(
        tiny_model, monkeypatch):
    """Catching KeyboardInterrupt leaves info as None, and the accept branch
    then falls back to x0 and returns the init point as if it were the MAP.

    That solution is pickled to map.pkl and reused by later runs, so an
    interrupted optimization becomes a silently wrong answer rather than a
    stopped one. The interrupt is injected after the objective has been
    evaluated, which is when a real ctrl-c would land.
    """
    evaluated = []

    def fake_minimize(objective, x0, **kwargs):
        objective(x0)
        evaluated.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(optim, 'minimize', fake_minimize)

    with pytest.raises(KeyboardInterrupt):
        optim.optimize(model=tiny_model, verbose=False)

    assert evaluated == [True], 'the interrupt must be raised after real work'


def test_hitting_the_evaluation_limit_still_returns_a_point(tiny_model, monkeypatch):
    """The control: StopIteration is the optimizer's own maxeval signal, not a
    user abort, so it must keep falling through to the best point so far."""
    def fake_minimize(objective, x0, **kwargs):
        raise StopIteration

    monkeypatch.setattr(optim, 'minimize', fake_minimize)

    point = optim.optimize(model=tiny_model, verbose=False)

    assert 'a' in point
    assert np.isfinite(point['a'])
