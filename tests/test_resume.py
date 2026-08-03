import logging
import os
import shutil

import numpy as np
import pytest
import yaml


SHORT = dict(tune=5, draws=5, chains=1, cores=1)


# ------------------------------------------------------- fast gate coverage

def _bare_fit(tmp_path, stale, model=None, map_soln=None):
    """A TransitFit carrying only what build_model reads.

    A real TransitFit needs data files and priors, and none of that is what
    these tests are about.
    """
    from timer import fit

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.clobber = False
    tf._stale_force_loaded = set(stale)
    tf._cache_keys = {'model': 'MODELKEY', 'run': 'RUNKEY'}
    tf.model = model
    tf.map_soln = map_soln
    tf.data, tf.priors, tf.masks = {}, {}, {}
    tf.nplanets, tf.use_gp, tf.chromatic = 1, False, False
    tf.fixed, tf.fit_basis = [], 'duration'
    tf.include_mean = True
    tf.include_flare = tf.chromatic_flare = False
    tf.include_bump = tf.chromatic_bump = False
    tf.use_custom_optimizer = True
    tf.gp_config = None
    return tf


class _FakeModel:
    """Stands in for a PyMC model, which is what build_model pickles."""


@pytest.fixture
def stub_build(monkeypatch):
    """Replaces model.build, recording whether it ran."""
    from timer import fit

    calls = []

    def fake_build(*args, **kwargs):
        calls.append(kwargs)
        return _FakeModel(), {'t0': np.array([0.07]), 'fresh': np.array(1.0)}

    monkeypatch.setattr(fit.model, 'build', fake_build)
    return calls


def test_build_model_does_not_reuse_a_stale_force_loaded_map(tmp_path, stub_build):
    """A map.pkl force-loaded past a key mismatch belongs to another config.

    It is present, so the plain `self.model is None` check reuses it and skips
    optimization; the run then samples from a MAP the current config never
    produced.
    """
    from timer import cache

    tf = _bare_fit(tmp_path, stale={'map.pkl'}, model=_FakeModel(),
                   map_soln={'cached': np.array(1.0)})

    tf.build_model(plot=False)

    assert len(stub_build) == 1, 'stale MAP was reused instead of re-optimized'
    assert 'fresh' in tf.map_soln
    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'map.pkl' not in manifest, (
        'a MAP written during a session that force-loaded a stale one must not '
        'be vouched for under the current key'
    )
    assert 'model.pkl' not in manifest


def test_build_model_does_not_reuse_a_stale_force_loaded_model(tmp_path, stub_build):
    """The model half of the same property: model.pkl carries the likelihood."""
    tf = _bare_fit(tmp_path, stale={'model.pkl'}, model=_FakeModel(),
                   map_soln={'cached': np.array(1.0)})

    tf.build_model(plot=False)

    assert len(stub_build) == 1, 'a stale pickled model was reused'


def test_build_model_still_reuses_a_matching_cached_model(tmp_path, stub_build):
    """The control: without a mismatch, reuse is the point of the cache."""
    from timer import cache

    tf = _bare_fit(tmp_path, stale=set(), model=_FakeModel(),
                   map_soln={'cached': np.array(1.0)})

    tf.build_model(plot=False)

    assert stub_build == [], 'a valid cached model must not be rebuilt'
    assert 'cached' in tf.map_soln
    assert cache.read_manifest(str(tmp_path)) is None, (
        'reusing a cached model writes nothing, so there is nothing to record'
    )


def test_build_model_rebuilds_when_only_the_map_is_missing(tmp_path, stub_build):
    """model.pkl and map.pkl are validated separately, so one can survive the
    other. Reusing a model without its MAP leaves map_soln unset and the run
    fails later, in clip_outliers, far from the cause.
    """
    tf = _bare_fit(tmp_path, stale=set(), model=_FakeModel(), map_soln=None)

    tf.build_model(plot=False)

    assert len(stub_build) == 1
    assert tf.map_soln is not None


def test_build_model_records_both_artifacts_under_the_model_key(tmp_path, stub_build):
    from timer import cache

    tf = _bare_fit(tmp_path, stale=set(), model=None, map_soln=None)

    tf.build_model(plot=False)

    manifest = cache.read_manifest(str(tmp_path))
    assert manifest['model.pkl'] == 'MODELKEY'
    assert manifest['map.pkl'] == 'MODELKEY'


def test_a_crash_during_the_rebuild_leaves_no_entry(tmp_path, monkeypatch):
    """An artifact stops being valid when the rebuild starts, not when it ends.

    clip_outliers records the new mask and then calls build_model(force=True)
    to refit on it. If that build dies, and the entries for the previous
    model.pkl/map.pkl are still standing, the next run reuses a model fitted
    to the unclipped data while _count_data reports the clipped total: the
    likelihood and every information criterion then disagree about how many
    points there are, permanently and with no warning.

    Ctrl-C is a live exit path here by design, so this is reachable without
    any hardware failure.
    """
    from timer import cache, fit

    cache.write_manifest(str(tmp_path), 'model.pkl', 'MODELKEY')
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fit.model, 'build', boom)
    tf = _bare_fit(tmp_path, stale=set(), model=_FakeModel(),
                   map_soln={'stale': np.array(1.0)})

    with pytest.raises(KeyboardInterrupt):
        tf.build_model(force=True, plot=False)

    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'model.pkl' not in manifest
    assert 'map.pkl' not in manifest


def test_a_stale_mask_disqualifies_the_model_built_on_it(tmp_path, stub_build):
    """mask.pkl feeds model.build, so a force-loaded stale mask contaminates
    the model and MAP just as surely as a stale map.pkl does.

    Reachable whenever the mask outlives the other artifacts: a config edit
    plus a deleted map.pkl, or a run that died before build_model finished.
    Recording the rebuilt pair under the current key would let the next
    ordinary run adopt a model fitted to a mask the config no longer produces,
    with no warning at all.
    """
    from timer import cache

    tf = _bare_fit(tmp_path, stale={'mask.pkl'}, model=None, map_soln=None)

    tf.build_model(plot=False)

    assert len(stub_build) == 1
    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'model.pkl' not in manifest
    assert 'map.pkl' not in manifest


def _bare_fit_for_clip(tmp_path, previous_mask, new_mask, monkeypatch):
    """A TransitFit carrying only what clip_outliers reads, with the mask the
    recompute will return stubbed out.

    clobber is set so the recompute happens even though a mask is already
    present, which is the from_dir-with-clobber path.
    """
    from timer import fit

    n = len(new_mask)
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.clobber = True
    tf._cache_keys = {'model': 'MODELKEY', 'run': 'RUNKEY'}
    tf._stale_force_loaded = set()
    tf.data = {'g': {'x': np.arange(float(n)), 'y': np.zeros(n)}}
    tf.fit_params = {'data': {'g': {'clip': True, 'clip_nsig': 5}}}
    tf.masks = {'g': previous_mask}
    tf.map_soln = {'g_light_curves': np.zeros(n)}
    tf.use_gp = tf.include_flare = tf.include_bump = False

    monkeypatch.setattr(fit.util, 'get_outlier_mask',
                        lambda *a, **k: new_mask.copy())
    rebuilt = []
    monkeypatch.setattr(fit.TransitFit, 'build_model',
                        lambda self, **kw: rebuilt.append(kw))
    return tf, rebuilt


def test_a_recomputed_mask_that_clips_less_still_refits(tmp_path, monkeypatch):
    """The model has to be refitted whenever the mask it was built on changed,
    not merely whenever the new mask excludes something.

    Reachable through from_dir with clobber: true, which loads a mask and then
    recomputes it. If the new mask excludes fewer points, n_outliers is 0 and
    the refit is skipped, so model.pkl and map.pkl keep describing the old
    masking while mask.pkl is rewritten and recorded under the current key.
    """
    previous = np.array([True, True, False, True])     # one point clipped
    fresh = np.array([True, True, True, True])         # nothing clipped now

    tf, rebuilt = _bare_fit_for_clip(tmp_path, previous, fresh, monkeypatch)
    tf.clip_outliers()

    assert rebuilt, 'the mask changed, so the model must be refitted'


def test_an_unchanged_recomputed_mask_does_not_refit(tmp_path, monkeypatch):
    """The control: recomputing the same mask must not trigger a pointless
    refit, which on a real fit costs the whole MAP optimization."""
    same = np.array([True, True, False, True])

    tf, rebuilt = _bare_fit_for_clip(tmp_path, same.copy(), same, monkeypatch)
    tf.clip_outliers()

    assert not rebuilt


def test_a_first_mask_with_no_outliers_does_not_refit(tmp_path, monkeypatch):
    """No previous mask and nothing clipped is not a change either."""
    tf, rebuilt = _bare_fit_for_clip(
        tmp_path, None, np.ones(4, dtype=bool), monkeypatch)
    tf.clip_outliers()

    assert not rebuilt


def test_a_first_mask_that_clips_refits(tmp_path, monkeypatch):
    tf, rebuilt = _bare_fit_for_clip(
        tmp_path, None, np.array([True, False, True, True]), monkeypatch)
    tf.clip_outliers()

    assert rebuilt


def _bare_fit_for_load(tmp_path, datasets):
    """A TransitFit carrying only what load_saved reads, over real data files."""
    from timer import fit

    for name in datasets:
        (tmp_path / f'{name}.csv').write_text(
            'time,flux,fluxerr\n0.0,1.0,0.001\n0.1,1.0,0.001\n')
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.wd = str(tmp_path)
    tf.outdir = str(tmp_path / 'out')
    tf.clobber = False
    tf._force_load_saved = True
    tf.fit_params = {
        'data': {n: {'file': f'{n}.csv', 'band': n} for n in datasets},
        'tune': 5, 'draws': 5, 'chains': 1, 'cores': 1, 'clobber': False,
    }
    tf.sys_params = {'star': {}, 'planets': {}}
    # the skeleton load_data leaves behind: one entry per dataset, all unclipped
    tf.masks = {n: None for n in datasets}
    tf.data = {n: {'x': np.zeros(2)} for n in datasets}
    tf.model = tf.map_soln = tf.trace = None
    return tf


def test_loading_a_mask_keeps_the_datasets_it_does_not_mention(tmp_path):
    """A mask.pkl from a run with fewer datasets must not replace the whole
    dict, or every later lookup by name raises KeyError.

    Adding a dataset to fit.yaml and opening the old output with from_dir is
    exactly the case from_dir exists to survive, and it currently dies in
    clip_outliers rather than warning.
    """
    from timer import fit

    tf = _bare_fit_for_load(tmp_path, ['g', 'r'])
    os.makedirs(tf.outdir, exist_ok=True)
    old = np.array([True, False])
    with open(os.path.join(tf.outdir, 'mask.pkl'), 'wb') as f:
        fit.pickle.dump({'g': old}, f)

    tf.load_saved()

    assert set(tf.masks) == {'g', 'r'}
    assert tf.masks['g'] is not None
    assert tf.masks['r'] is None


def test_loading_a_mask_ignores_datasets_the_config_no_longer_has(tmp_path):
    """Removing a dataset from fit.yaml leaves its mask in mask.pkl. Looking
    that name up in self.data to length check it raises KeyError, so the
    unknown entry has to be skipped before anything else touches it."""
    from timer import fit

    tf = _bare_fit_for_load(tmp_path, ['g'])
    os.makedirs(tf.outdir, exist_ok=True)
    with open(os.path.join(tf.outdir, 'mask.pkl'), 'wb') as f:
        fit.pickle.dump({'g': np.array([True, False]),
                         'z': np.array([True, True])}, f)

    tf.load_saved()

    assert set(tf.masks) == {'g'}


def test_loading_a_mask_drops_entries_whose_length_no_longer_matches(tmp_path):
    """A binsize or trim edit changes how many points a dataset has. A mask of
    the old length silently misaligns with the data instead of failing, so it
    has to be discarded rather than adopted."""
    from timer import fit

    tf = _bare_fit_for_load(tmp_path, ['g'])
    os.makedirs(tf.outdir, exist_ok=True)
    with open(os.path.join(tf.outdir, 'mask.pkl'), 'wb') as f:
        fit.pickle.dump({'g': np.ones(7, dtype=bool)}, f)   # data has 2 points

    tf.load_saved()

    assert tf.masks['g'] is None


def test_a_crash_while_overwriting_an_artifact_leaves_no_entry(
        tmp_path, stub_build, monkeypatch):
    """Why the manifest entry is dropped before the file is rewritten.

    A previous run recorded model.pkl under this key. If the process dies part
    way through overwriting it, the surviving entry would vouch for a half
    written file and the next run would load it without a warning.
    """
    from timer import cache, fit

    cache.write_manifest(str(tmp_path), 'model.pkl', 'MODELKEY')
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')

    def boom(*args, **kwargs):
        raise OSError('no space left on device')

    monkeypatch.setattr(fit.pickle, 'dump', boom)
    tf = _bare_fit(tmp_path, stale=set(), model=None, map_soln=None)

    with pytest.raises(OSError):
        tf.build_model(plot=False)

    manifest = cache.read_manifest(str(tmp_path))
    assert 'model.pkl' not in manifest


# ------------------------------------------------------ real resume coverage

def _load_params(wd):
    with open(os.path.join(wd, 'fit.yaml')) as f:
        fit_params = yaml.safe_load(f)
    with open(os.path.join(wd, 'sys.yaml')) as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(SHORT)
    fit_params['clobber'] = False
    return fit_params, sys_params


def _write_fit_yaml(wd, fit_params):
    """from_dir re-reads fit.yaml, so an edit only counts once it is on disk."""
    with open(os.path.join(wd, 'fit.yaml'), 'w') as f:
        yaml.safe_dump(fit_params, f)


def _run(wd, fit_params, sys_params):
    from timer import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)
    return tf


@pytest.fixture(scope='module')
def baseline(tmp_path_factory, make_project_module):
    """One real fit, copied per test so each gets an isolated directory."""
    wd = tmp_path_factory.mktemp('baseline') / 'proj'
    make_project_module(str(wd))
    fit_params, sys_params = _load_params(wd)
    _run(wd, fit_params, sys_params)
    return wd


@pytest.fixture
def wd(baseline, tmp_path):
    target = tmp_path / 'proj'
    shutil.copytree(baseline, target)
    return target


@pytest.mark.slow
def test_unchanged_rerun_reuses_everything(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'reusing cached model' in caplog.text
    assert 'sampling for' not in caplog.text


@pytest.mark.slow
def test_config_edit_recomputes_model_and_trace(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    fit_params['data']['g']['trend'] = 2
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_draws_bump_reuses_the_model_and_resamples(wd, caplog):
    """The property the two tier split exists for."""
    fit_params, sys_params = _load_params(wd)
    fit_params['draws'] = SHORT['draws'] + 1
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'reusing cached model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_a_new_seed_refits_the_model_and_resamples(wd, caplog):
    """The counterpart of the draws bump: a seed is not a sampler-only knob.

    It is passed to numpy's global RNG before the limb darkening priors are
    drawn, so it moves the priors the MAP is optimized against. Reusing the
    cached model here would start the chain from a MAP belonging to a different
    prior set, and derive *-cor.csv and the GP edf from it.
    """
    fit_params, sys_params = _load_params(wd)
    fit_params['random_seed'] = fit_params['random_seed'] + 1
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_missing_manifest_recomputes_everything(wd, caplog):
    """Every output directory that predates this feature is in this state."""
    from timer import cache
    os.remove(os.path.join(wd, 'out', cache.MANIFEST_NAME))
    fit_params, sys_params = _load_params(wd)
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_data_edit_recomputes_the_model(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    target = wd / fit_params['data']['g']['file']
    # a trailing blank line: guaranteed byte level change, and pandas skips
    # blank lines, so the parsed data is identical and only the hash moves
    target.write_text(target.read_text() + '\n')
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text


@pytest.mark.slow
def test_from_dir_loads_mismatched_artifacts_with_a_warning(wd, caplog):
    """from_dir is an explicit request for the artifacts in a directory.

    Loading a finished run for plotting must still work after the config has
    moved on, so a mismatch warns rather than skipping. This is the only
    behavior that distinguishes _force_load_saved.
    """
    from timer import fit

    fit_params, _ = _load_params(wd)
    fit_params['data']['g']['trend'] = 2
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))

    assert 'does not match the current config' in caplog.text
    assert 'loading map.pkl anyway' in caplog.text
    assert tf.trace is not None
    assert tf.map_soln is not None


@pytest.mark.slow
def test_from_dir_build_model_does_not_launder_a_stale_map(wd, caplog):
    """A model-tier edit invalidates map.pkl; from_dir loads it anyway.

    Reusing it would sample from a MAP the current config never produced, and
    recording it under the current key would make the next ordinary run adopt
    it with no warning at all.
    """
    from timer import cache, fit

    fit_params, _ = _load_params(wd)
    fit_params['data']['g']['trend'] = 2
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.build_model(verbose=False, plot=False)

    assert 'loading map.pkl anyway' in caplog.text, 'setup: the MAP must be force-loaded stale'
    assert 'building and optimizing model' in caplog.text, 'a stale MAP was reused'

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert 'map.pkl' not in manifest, 'the safe outcome is no entry, so the next run recomputes'
    assert 'model.pkl' not in manifest


@pytest.mark.slow
def test_from_dir_run_tier_edit_resamples_and_records_nothing_derived(wd, caplog):
    """A stale trace must be treated as absent, not silently resumed from.

    Bumping draws only changes the run key, so from_dir force-loads the
    mismatched trace with a warning. If sample() then trusts self.trace it
    skips MCMC entirely and rederives summary.csv and map.pkl from the old
    draws; because map.pkl itself was never flagged it would then be recorded
    under the current key.
    """
    from timer import cache, fit

    fit_params, _ = _load_params(wd)
    fit_params['draws'] = SHORT['draws'] + 1
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.build_model(verbose=False, plot=False)
        tf.sample(plot_fit=False, plot_systematics=False)

    assert 'loading trace.pkl anyway' in caplog.text, 'setup: the trace must be force-loaded stale'
    assert 'reusing cached model' in caplog.text, (
        'setup: only the run key may have moved, so the model is still valid'
    )
    assert 'sampling for' in caplog.text, 'MCMC was skipped on a stale trace'

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert 'trace.pkl' not in manifest, 'a force-loaded stale trace was re-recorded as valid'
    assert 'map.pkl' not in manifest, (
        'map.pkl is derived from the trace, so it may not be recorded either'
    )


@pytest.mark.slow
def test_from_dir_sample_records_a_freshly_computed_trace(wd, caplog):
    """The control for the test above: with nothing stale to force-load, a
    rerun records both artifacts, so those assertions are not vacuous."""
    from timer import cache, fit

    os.remove(os.path.join(wd, 'out', 'trace.pkl'))
    fit_params, _ = _load_params(wd)
    _write_fit_yaml(wd, fit_params)

    tf = fit.TransitFit.from_dir(str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest['trace.pkl'] == tf._cache_keys['run']
    assert manifest['map.pkl'] == tf._cache_keys['model']
