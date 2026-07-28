import time

from gen2_architectures.training.data_prep import _cheap_file_identity


def test_identity_changes_when_file_content_and_mtime_change(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    path.write_bytes(b"v1")
    first = _cheap_file_identity(path)

    time.sleep(0.01)
    path.write_bytes(b"v2-longer-content")
    second = _cheap_file_identity(path)

    assert first != second


def test_identity_is_stable_for_an_unchanged_file(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    path.write_bytes(b"same")
    assert _cheap_file_identity(path) == _cheap_file_identity(path)


def test_missing_file_does_not_raise():
    identity = _cheap_file_identity("/no/such/path/here.ckpt")
    assert "no/such/path" in identity
