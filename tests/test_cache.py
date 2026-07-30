import copy
import json
import os

import pytest

from timer import cache


@pytest.fixture
def project(tmp_path):
    """A working directory with two data files, plus matching config dicts."""
    (tmp_path / 'g.csv').write_text('time,flux,fluxerr\n0.0,1.0,0.001\n0.1,1.0,0.001\n')
    (tmp_path / 'r.csv').write_text('time,flux,fluxerr\n0.0,1.0,0.002\n0.1,1.0,0.002\n')
    fit_params = {
        'data': {
            'g': {'file': 'g.csv', 'band': 'g', 'binsize': None},
            'r': {'file': 'r.csv', 'band': 'r', 'binsize': None},
        },
        'planets': 'c',
        'fixed': ['period'],
        'use_gp': False,
        'tune': 100, 'draws': 200, 'chains': 2, 'cores': 2, 'clobber': False,
    }
    sys_params = {'star': {'teff': [5675, 75]}, 'planets': {'c': {'period': [14.3, 1e-5]}}}
    return str(tmp_path), fit_params, sys_params


def keys_for(project, **edits):
    wd, fit_params, sys_params = project
    fit_params = copy.deepcopy(fit_params)
    fit_params.update(edits)
    return cache.compute_keys(fit_params, sys_params, wd)


@pytest.mark.parametrize('setting,value', [
    ('fixed', ['period', 'b']),
    ('use_gp', True),
    ('chromatic', True),
    ('include_mean', False),
    ('planets', 'b'),
    ('uniform', {'ror': [0.01, 0.15]}),
])
def test_model_key_changes_when_a_model_setting_changes(project, setting, value):
    """Editing fit.yaml and rerunning must not silently reuse the old MAP.

    Parametrized over several settings because a single one only proves that
    that one setting is in the model tier: misfiling any other under NO_EFFECT
    would go unnoticed.
    """
    before = keys_for(project)
    after = keys_for(project, **{setting: value})
    assert after['model'] != before['model']


def test_model_key_changes_when_a_per_dataset_setting_changes(project):
    """Data options live one level down, inside fit_params['data'][name], so
    they only reach the key if the whole nested structure is hashed."""
    wd, fit_params, sys_params = project
    edited = copy.deepcopy(fit_params)
    edited['data']['g']['trend'] = 1
    assert cache.compute_keys(edited, sys_params, wd)['model'] != \
        cache.compute_keys(fit_params, sys_params, wd)['model']


def test_model_key_changes_when_two_datasets_swap_files(project):
    """Which file each dataset reads is part of the model. Hashing only the
    file contents, and dropping the config that maps names to files, would
    leave the two arrangements indistinguishable."""
    wd, fit_params, sys_params = project
    swapped = copy.deepcopy(fit_params)
    swapped['data']['g']['file'] = 'r.csv'
    swapped['data']['r']['file'] = 'g.csv'
    assert cache.compute_keys(swapped, sys_params, wd)['model'] != \
        cache.compute_keys(fit_params, sys_params, wd)['model']


def test_model_key_changes_when_the_system_parameters_change(project):
    """sys.yaml sets the priors, so it belongs in the model tier too."""
    wd, fit_params, sys_params = project
    edited = copy.deepcopy(sys_params)
    edited['star']['teff'] = [5800, 75]
    assert cache.compute_keys(fit_params, edited, wd)['model'] != \
        cache.compute_keys(fit_params, sys_params, wd)['model']


def test_model_key_changes_when_the_data_bytes_change(project):
    """Reprocessing a light curve must invalidate everything downstream, even
    though nothing in the config moved."""
    wd, fit_params, sys_params = project
    before = cache.compute_keys(fit_params, sys_params, wd)['model']
    with open(os.path.join(wd, 'g.csv'), 'a') as f:
        f.write('0.2,1.0,0.001\n')
    assert cache.compute_keys(fit_params, sys_params, wd)['model'] != before


def test_model_key_survives_moving_the_project_directory(tmp_path, project):
    """Data is hashed by content and keyed by dataset name, so copying an
    example elsewhere must not throw away a cache whose inputs are identical."""
    wd, fit_params, sys_params = project
    import shutil
    other = tmp_path / 'elsewhere'
    shutil.copytree(wd, other)
    assert cache.compute_keys(fit_params, sys_params, str(other)) == \
        cache.compute_keys(fit_params, sys_params, wd)


def test_model_key_changes_when_a_dataset_is_renamed(project):
    """The dataset name reaches every site name in the model, so a rename is a
    different model even with byte identical data."""
    wd, fit_params, sys_params = project
    renamed = copy.deepcopy(fit_params)
    renamed['data']['g2'] = renamed['data'].pop('g')
    assert cache.compute_keys(renamed, sys_params, wd)['model'] != \
        cache.compute_keys(fit_params, sys_params, wd)['model']


@pytest.mark.parametrize('setting,value', [('tune', 101), ('draws', 201), ('chains', 3)])
def test_sampler_settings_move_the_run_key_only(project, setting, value):
    """The whole point of two tiers: more draws must resample without paying
    for a fresh MAP optimization."""
    before = keys_for(project)
    after = keys_for(project, **{setting: value})
    assert after['model'] == before['model']
    assert after['run'] != before['run']


def test_a_model_edit_also_moves_the_run_key(project):
    """A trace produced under a different model is invalid, so the run key has
    to be derived from the model key rather than from the sampler alone."""
    before = keys_for(project)
    after = keys_for(project, fixed=['period', 'b'])
    assert after['run'] != before['run']


@pytest.mark.parametrize('setting,value', [('cores', 4), ('clobber', True)])
def test_settings_with_no_effect_on_results_move_neither_key(project, setting, value):
    """Running on more cores, or passing clobber, changes nothing about what
    the answer is."""
    assert keys_for(project, **{setting: value}) == keys_for(project)


def test_keys_do_not_depend_on_dict_ordering(project):
    """yaml.load ordering must not decide whether a cache is adopted."""
    wd, fit_params, sys_params = project
    reordered = {k: fit_params[k] for k in reversed(list(fit_params))}
    assert cache.compute_keys(reordered, sys_params, wd) == \
        cache.compute_keys(fit_params, sys_params, wd)


def test_an_unknown_setting_lands_in_the_model_tier(project):
    """Options added later must invalidate more rather than less, so anything
    not explicitly classified counts as part of the model."""
    before = keys_for(project)
    after = keys_for(project, some_future_option=True)
    assert after['model'] != before['model']


# ------------------------------------------------------------------ manifest

def test_read_manifest_returns_none_when_there_is_no_manifest(tmp_path):
    assert cache.read_manifest(str(tmp_path)) is None


def test_read_manifest_returns_none_for_unparseable_json(tmp_path):
    """A manifest truncated by a crash must read as absent, not raise."""
    (tmp_path / cache.MANIFEST_NAME).write_text('{"format_version": 1,')
    assert cache.read_manifest(str(tmp_path)) is None


def test_a_version_1_manifest_is_rejected(tmp_path):
    """Caches written before the binning and clipping fixes must not validate.

    bin_df now scales median-binned errors by sqrt(pi/2) and get_outlier_mask
    scales the clip threshold by 1.4826, so both the likelihood weights and the
    mask changed. compute_keys digests the config and the data file bytes,
    neither of which moved, so an out/ directory from before those fixes would
    otherwise read as current: map.pkl reused, MCMC skipped, and results written
    from a posterior fitted to errors 25 percent too small, permanently until
    clobber. Bumping the format version is the only thing that invalidates it.
    """
    (tmp_path / cache.MANIFEST_NAME).write_text(
        json.dumps({'format_version': 1, 'map.pkl': 'K', 'trace.pkl': 'R'}))
    assert cache.read_manifest(str(tmp_path)) is None


def test_read_manifest_returns_none_for_a_foreign_format_version(tmp_path):
    """This is what makes bumping FORMAT_VERSION invalidate caches on disk."""
    (tmp_path / cache.MANIFEST_NAME).write_text(
        json.dumps({'format_version': cache.FORMAT_VERSION + 1, 'map.pkl': 'K'}))
    assert cache.read_manifest(str(tmp_path)) is None


def test_write_manifest_preserves_other_entries(tmp_path):
    """map.pkl and trace.pkl are recorded at different times, so writing one
    must not forget the other."""
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')
    cache.write_manifest(str(tmp_path), 'trace.pkl', 'RUNKEY')
    manifest = cache.read_manifest(str(tmp_path))
    assert manifest['map.pkl'] == 'MODELKEY'
    assert manifest['trace.pkl'] == 'RUNKEY'


def test_is_valid_only_accepts_the_recorded_key(tmp_path):
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')
    manifest = cache.read_manifest(str(tmp_path))
    assert cache.is_valid(manifest, 'map.pkl', 'MODELKEY')
    assert not cache.is_valid(manifest, 'map.pkl', 'OTHERKEY')
    assert not cache.is_valid(manifest, 'trace.pkl', 'MODELKEY')
    assert not cache.is_valid(None, 'map.pkl', 'MODELKEY')


def test_drop_entry_forgets_only_the_named_artifact(tmp_path):
    """Dropping before overwriting is what makes a crash mid-write leave no
    entry rather than an entry vouching for a half written file."""
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')
    cache.write_manifest(str(tmp_path), 'trace.pkl', 'RUNKEY')
    cache.drop_entry(str(tmp_path), 'map.pkl')
    manifest = cache.read_manifest(str(tmp_path))
    assert 'map.pkl' not in manifest
    assert manifest['trace.pkl'] == 'RUNKEY'


def test_drop_entry_is_a_no_op_without_a_manifest(tmp_path):
    """Every write site drops first, including the very first one."""
    cache.drop_entry(str(tmp_path), 'map.pkl')
    assert cache.read_manifest(str(tmp_path)) is None


def test_a_failed_manifest_write_leaves_the_previous_one_intact(tmp_path, monkeypatch):
    """Opening the manifest for writing truncates it before anything is
    serialized, so a crash in that window destroys every entry, including a
    trace.pkl that cost hours. This whole module exists for crash safety, so
    it cannot itself have a window where a crash loses data.
    """
    cache.write_manifest(str(tmp_path), 'trace.pkl', 'RUNKEY')

    def boom(*args, **kwargs):
        raise OSError('no space left on device')

    monkeypatch.setattr(cache.json, 'dump', boom)
    with pytest.raises(OSError):
        cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')

    manifest = cache.read_manifest(str(tmp_path))
    assert manifest is not None, 'the previous manifest was destroyed'
    assert manifest['trace.pkl'] == 'RUNKEY'


def test_a_failed_drop_leaves_the_previous_manifest_intact(tmp_path, monkeypatch):
    cache.write_manifest(str(tmp_path), 'trace.pkl', 'RUNKEY')
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')

    def boom(*args, **kwargs):
        raise OSError('no space left on device')

    monkeypatch.setattr(cache.json, 'dump', boom)
    with pytest.raises(OSError):
        cache.drop_entry(str(tmp_path), 'map.pkl')

    manifest = cache.read_manifest(str(tmp_path))
    assert manifest is not None
    assert manifest['trace.pkl'] == 'RUNKEY'
