"""Cache keys tying resume artifacts to the config and data that produced them.

Imports only the standard library so this logic stays testable without
building a model.
"""
import hashlib
import json
import os

FORMAT_VERSION = 1

# keys that affect sampling only, never the model or the MAP solution
RUN_TIER = frozenset({'tune', 'draws', 'chains'})
# keys that affect neither the model nor the results
NO_EFFECT = frozenset({'cores', 'clobber'})


def _digest(obj):
    """Stable sha256 of a json serializable object, insensitive to dict ordering."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def hash_data_files(fit_params, wd):
    """sha256 of each dataset's file contents, keyed by dataset name.

    Keyed by name rather than by path so relocating a project directory, or
    copying an example into a temporary directory, does not invalidate a cache
    whose data is byte identical. Dataset names are themselves part of
    fit_params, so renaming or adding a dataset still invalidates.
    """
    digests = {}
    for name, spec in fit_params['data'].items():
        h = hashlib.sha256()
        with open(os.path.join(wd, spec['file']), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        digests[name] = h.hexdigest()
    return digests


def compute_keys(fit_params, sys_params, wd):
    """Return {'model': ..., 'run': ...} for the current config and data.

    Anything not named in RUN_TIER or NO_EFFECT falls into the model tier, so
    an option added later invalidates more rather than less.
    """
    model_params = {k: v for k, v in fit_params.items()
                    if k not in RUN_TIER and k not in NO_EFFECT}
    model_key = _digest({
        'format_version': FORMAT_VERSION,
        'fit': model_params,
        'sys': sys_params,
        'data': hash_data_files(fit_params, wd),
    })
    run_key = _digest({
        'model': model_key,
        'sampler': {k: fit_params.get(k) for k in sorted(RUN_TIER)},
    })
    return {'model': model_key, 'run': run_key}


MANIFEST_NAME = 'cache.json'


def manifest_path(outdir):
    return os.path.join(outdir, MANIFEST_NAME)


def read_manifest(outdir):
    """Parsed manifest, or None if absent, unreadable, or a foreign format version.

    Returning None for a version mismatch is what makes bumping FORMAT_VERSION
    invalidate every cache already on disk.
    """
    fp = manifest_path(outdir)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp) as f:
            manifest = json.load(f)
    except (ValueError, OSError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get('format_version') != FORMAT_VERSION:
        return None
    return manifest


def write_manifest(outdir, artifact, key):
    """Record that `artifact` was written under `key`, preserving other entries."""
    manifest = read_manifest(outdir) or {'format_version': FORMAT_VERSION}
    manifest[artifact] = key
    with open(manifest_path(outdir), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def is_valid(manifest, artifact, expected_key):
    """True when the manifest records `artifact` as written under `expected_key`."""
    if not manifest:
        return False
    return manifest.get(artifact) == expected_key


def drop_entry(outdir, artifact):
    """Forget which key `artifact` was written under.

    Called immediately before overwriting an artifact. If the process dies
    between that write and the following write_manifest, no entry vouches for
    the new file, so it reads as stale and is recomputed rather than being
    loaded under the previous config's key.
    """
    manifest = read_manifest(outdir)
    if not manifest or artifact not in manifest:
        return
    del manifest[artifact]
    with open(manifest_path(outdir), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
