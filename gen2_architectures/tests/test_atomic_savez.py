import tempfile
from pathlib import Path

import numpy as np

from gen2_architectures.training.data_prep import _atomic_savez


def test_atomic_savez_writes_readable_file_at_the_exact_requested_path():
    """Regression test: np.savez SILENTLY APPENDS '.npz' to any string/
    Path target that doesn't already end in '.npz' (a well-known numpy
    gotcha). The tmp path used to be built as cache_path.with_suffix(
    cache_path.suffix + f'.tmp{pid}') -- e.g. 'INT1.npz.tmp123', which
    does NOT end in '.npz' -- so numpy actually wrote
    'INT1.npz.tmp123.npz', and the following os.replace() raised
    FileNotFoundError looking for a path numpy never created. Hit for
    real on the training server on the first cold (non-cached) GigaPath
    feature computation through this module."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "INT1.npz"
        _atomic_savez(cache_path, features=np.arange(12).reshape(4, 3))
        assert cache_path.is_file()
        loaded = np.load(cache_path)
        assert np.array_equal(loaded["features"], np.arange(12).reshape(4, 3))
        # no stray "<name>.npz.tmp<pid>" (or "<name>.npz.tmp<pid>.npz")
        # left behind in the target directory
        leftovers = [p for p in Path(tmp).iterdir() if p != cache_path]
        assert leftovers == [], f"unexpected leftover files: {leftovers}"


def test_atomic_savez_can_be_called_repeatedly_for_the_same_path():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "sample.npz"
        _atomic_savez(cache_path, x=np.array([1, 2, 3]))
        _atomic_savez(cache_path, x=np.array([4, 5, 6]))
        loaded = np.load(cache_path)
        assert np.array_equal(loaded["x"], np.array([4, 5, 6]))
