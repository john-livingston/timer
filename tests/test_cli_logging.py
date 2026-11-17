"""What a command-line run prints without -v.

The log file gets everything. The terminal used to get nothing at all unless
-v was given, so a warning that the MAP optimizer had fallen back to a weaker
method was only visible to someone who opened the log.
"""
import logging

import pytest

from timer import fit


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_a_quiet_run_prints_warnings_but_not_info(tmp_path, capsys, restore_root_logger):
    fit.setup_logging(str(tmp_path), verbose=False)
    logging.info('routine progress line')
    logging.warning('optimizer fell back')
    err = capsys.readouterr().err
    assert 'optimizer fell back' in err
    assert 'routine progress line' not in err


def test_a_verbose_run_prints_info_once(tmp_path, capsys, restore_root_logger):
    fit.setup_logging(str(tmp_path), verbose=True)
    logging.info('routine progress line')
    logging.warning('optimizer fell back')
    err = capsys.readouterr().err
    assert err.count('routine progress line') == 1
    assert err.count('optimizer fell back') == 1
