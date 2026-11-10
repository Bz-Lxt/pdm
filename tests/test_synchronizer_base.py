from __future__ import annotations

import base64
import dataclasses
import hashlib
from pathlib import Path

import pytest

from pdm.exceptions import BuildError
from pdm.installers.base import BaseSynchronizer, editables_candidate, is_distribution_damaged
from pdm.models.candidates import Candidate
from pdm.models.requirements import parse_requirement


def make_synchronizer(mocker):
    synchronizer = object.__new__(BaseSynchronizer)
    synchronizer.environment = mocker.Mock()
    synchronizer.environment.project.backend.expand_line.side_effect = lambda value: value
    synchronizer.reinstall = False
    synchronizer.no_editable = False
    synchronizer.install_self = False
    return synchronizer


def _record_hash(content: bytes, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm, content).digest()
    return algorithm + "=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def make_record_dist(mocker, tmp_path: Path, files: dict[str, bytes], *, extra_rows=(), editable=False, texts=None):
    """Build a mock distribution backed by real files with a matching RECORD."""
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    record_rows = [f"{rel},{_record_hash(content)},{len(content)}" for rel, content in files.items()]
    record_rows.extend(extra_rows)
    record_rows.append("foo-1.0.dist-info/RECORD,,")
    stored_texts = {"RECORD": "\n".join(record_rows) + "\n"}
    if texts:
        stored_texts.update(texts)
    dist = mocker.Mock()
    dist.read_text.side_effect = lambda name: stored_texts.get(name)
    dist.locate_file.side_effect = lambda rel: tmp_path / rel
    mocker.patch("pdm.installers.base.is_editable", return_value=editable)
    return dist


@pytest.mark.parametrize("has_match", [False, True])
def test_editables_candidate(mocker, has_match):
    best = mocker.Mock() if has_match else None
    finder = mocker.MagicMock()
    finder.find_best_match.return_value.best = best
    environment = mocker.MagicMock()
    environment.get_finder.return_value.__enter__.return_value = finder
    from_installation_candidate = mocker.patch(
        "pdm.installers.base.Candidate.from_installation_candidate",
    )

    result = editables_candidate(environment)

    if has_match:
        assert result is from_installation_candidate.return_value
        from_installation_candidate.assert_called_once()
    else:
        assert result is None


@pytest.mark.parametrize(
    ("install_self", "has_editables", "requires", "build_error", "expected"),
    [
        (False, False, ["editables"], False, False),
        (True, True, ["editables"], False, False),
        (True, False, ["other"], False, False),
        (True, False, ["editables>=0.5"], False, True),
        (True, False, [], True, False),
    ],
)
def test_should_install_editables(mocker, install_self, has_editables, requires, build_error, expected):
    synchronizer = make_synchronizer(mocker)
    synchronizer.install_self = install_self
    synchronizer.requested_candidates = {"editables": mocker.Mock()} if has_editables else {}
    prepared = mocker.Mock()
    prepared.metadata.requires = requires
    candidate = mocker.Mock()
    candidate.prepare.return_value = prepared
    if build_error:
        candidate.prepare.side_effect = BuildError("failed")
    synchronizer.__dict__["self_candidate"] = candidate

    assert synchronizer.should_install_editables() is expected


def test_candidates_marks_requested_and_converts_editable(mocker):
    synchronizer = make_synchronizer(mocker)
    editable_req = dataclasses.replace(parse_requirement("demo"), editable=True)
    editable = Candidate(editable_req, name="demo", version="1.0")
    regular = Candidate(parse_requirement("regular"), name="regular", version="2.0")
    synchronizer.requested_candidates = {"demo": editable, "regular": regular}
    synchronizer.requirements = [parse_requirement("regular")]
    synchronizer.no_editable = ["demo"]

    result = synchronizer.candidates

    assert result["demo"].req.editable is False
    assert result["regular"].requested is True


def test_candidates_adds_editables_dependency(mocker):
    synchronizer = make_synchronizer(mocker)
    synchronizer.requested_candidates = {}
    synchronizer.requirements = []
    synchronizer.environment.project.all_dependencies = {}
    synchronizer.no_editable = False
    mocker.patch.object(synchronizer, "should_install_editables", return_value=True)
    candidate = mocker.patch("pdm.installers.base.editables_candidate").return_value

    assert synchronizer.candidates == {"editables": candidate}


def test_should_update_immediate_cases(mocker):
    synchronizer = make_synchronizer(mocker)
    dist = mocker.Mock()
    candidate = mocker.Mock()
    candidate.req.editable = False

    synchronizer.reinstall = True
    assert synchronizer._should_update(dist, candidate)
    synchronizer.reinstall = False
    candidate.req.editable = True
    assert synchronizer._should_update(dist, candidate)
    candidate.req.editable = False
    mocker.patch("pdm.installers.base.is_editable", return_value=True)
    assert not synchronizer._should_update(dist, candidate)
    synchronizer.no_editable = True
    assert synchronizer._should_update(dist, candidate)


@pytest.mark.parametrize(
    ("direct_url", "hashes", "expected"),
    [
        (None, [], False),
        ('{"url": "x"}', [], False),
        ('{"archive_info": {"hash": "sha256=abc"}}', [{"hash": "sha256:abc"}], False),
        ('{"archive_info": {"hash": "sha256=abc"}}', [{"hash": "sha256:def"}], True),
    ],
)
def test_should_update_archive_requirement(mocker, direct_url, hashes, expected):
    synchronizer = make_synchronizer(mocker)
    mocker.patch("pdm.installers.base.is_editable", return_value=False)
    dist = mocker.Mock()
    dist.read_text.return_value = direct_url
    installed_req = parse_requirement("demo @ https://example.org/demo.whl")
    mocker.patch("pdm.installers.base.Requirement.from_dist", return_value=installed_req)
    candidate = mocker.Mock()
    candidate.req.is_named = False
    candidate.req.editable = False
    candidate.link.url_without_fragment = "https://example.org/demo.whl"
    candidate.hashes = hashes

    assert synchronizer._should_update(dist, candidate) is expected


def test_should_update_changed_url_and_local_directory(mocker):
    synchronizer = make_synchronizer(mocker)
    mocker.patch("pdm.installers.base.is_editable", return_value=False)
    candidate = mocker.Mock()
    candidate.req.is_named = False
    candidate.req.editable = False
    candidate.link.url_without_fragment = "https://example.org/new.whl"
    installed_req = mocker.Mock()
    installed_req.is_local_dir = False
    installed_req.get_full_url.return_value = "https://example.org/old.whl"
    mocker.patch("pdm.installers.base.Requirement.from_dist", return_value=installed_req)
    mocker.patch("pdm.installers.base.FileRequirement", type(installed_req))

    assert synchronizer._should_update(mocker.Mock(), candidate)
    installed_req.is_local_dir = True
    assert synchronizer._should_update(mocker.Mock(), candidate)


def test_should_update_unknown_direct_requirement(mocker):
    synchronizer = make_synchronizer(mocker)
    mocker.patch("pdm.installers.base.is_editable", return_value=False)
    mocker.patch("pdm.installers.base.Requirement.from_dist", return_value=mocker.Mock())
    candidate = mocker.Mock()
    candidate.req.is_named = False
    candidate.req.editable = False

    assert synchronizer._should_update(mocker.Mock(), candidate)


def test_self_key_preserves_empty_project_name(mocker):
    synchronizer = make_synchronizer(mocker)
    synchronizer.install_self = True
    synchronizer.environment.project.name = ""

    assert synchronizer.self_key == ""


def test_synchronize_applies_all_operations(mocker):
    synchronizer = make_synchronizer(mocker)
    synchronizer.compare_with_working_set = mocker.Mock(return_value=(["add"], ["update"], ["remove"]))
    add_candidate = mocker.Mock(version="1")
    update_candidate = mocker.Mock(version="2")
    synchronizer.__dict__["candidates"] = {"add": add_candidate, "update": update_candidate}
    update_dist = mocker.Mock(version="1")
    remove_dist = mocker.Mock(version="3")
    synchronizer.working_set = {"update": update_dist, "remove": remove_dist}
    manager = mocker.Mock()
    synchronizer.get_manager = mocker.Mock(return_value=manager)

    synchronizer.synchronize()

    manager.install.assert_called_once_with(add_candidate)
    manager.overwrite.assert_called_once_with(update_dist, update_candidate)
    manager.uninstall.assert_called_once_with(remove_dist)


@pytest.mark.parametrize("already_installed", [False, True])
def test_synchronize_installs_self(mocker, already_installed):
    synchronizer = make_synchronizer(mocker)
    synchronizer.install_self = True
    synchronizer.no_editable = False
    synchronizer.compare_with_working_set = mocker.Mock(return_value=([], [], []))
    synchronizer.__dict__["self_candidate"] = mocker.sentinel.self_candidate
    synchronizer.environment.project.name = "Demo"
    dist = mocker.sentinel.dist
    synchronizer.working_set = {"demo": dist} if already_installed else {}
    manager = mocker.Mock()
    synchronizer.get_manager = mocker.Mock(return_value=manager)

    synchronizer.synchronize()

    if already_installed:
        manager.overwrite.assert_called_once_with(dist, mocker.sentinel.self_candidate)
    else:
        manager.install.assert_called_once_with(mocker.sentinel.self_candidate)


def test_damaged_detection_accepts_intact_distribution(mocker, tmp_path):
    dist = make_record_dist(
        mocker, tmp_path, {"demo.py": b"print('hi')\n", "demo-1.0.dist-info/METADATA": b"Name: demo\n"}
    )
    assert is_distribution_damaged(dist) is False


def test_damaged_detection_catches_missing_file(mocker, tmp_path):
    dist = make_record_dist(mocker, tmp_path, {"demo.py": b"print('hi')\n"})
    (tmp_path / "demo.py").unlink()
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_catches_same_size_modification(mocker, tmp_path):
    dist = make_record_dist(mocker, tmp_path, {"demo.py": b"print('hi')\n"})
    (tmp_path / "demo.py").write_bytes(b"print('ho')\n")
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_catches_size_mismatch(mocker, tmp_path):
    dist = make_record_dist(mocker, tmp_path, {"demo.py": b"print('hi')\n"})
    (tmp_path / "demo.py").write_bytes(b"x")
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_ignores_unrecorded_changes(mocker, tmp_path):
    # Rows without hash nor size cannot be verified, missing files must not trigger.
    dist = make_record_dist(mocker, tmp_path, {"demo.py": b"x\n"}, extra_rows=["ghost.txt,,"])
    assert is_distribution_damaged(dist) is False


@pytest.mark.parametrize("hash_field", ["", "sha256="])
def test_damaged_detection_link_entry_without_hash_uses_size(mocker, tmp_path, hash_field):
    # Symlink/hardlink cache installs record no digest, only the size.
    dist = make_record_dist(mocker, tmp_path, {}, extra_rows=[f"linked.py,{hash_field},5"])
    target = tmp_path / "linked.py"
    target.write_bytes(b"12345")
    assert is_distribution_damaged(dist) is False
    # A same-size content drift cannot be detected without a recorded hash.
    target.write_bytes(b"54321")
    assert is_distribution_damaged(dist) is False
    target.write_bytes(b"1234")
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_catches_dangling_symlink(mocker, tmp_path):
    link = tmp_path / "link.py"
    try:
        link.symlink_to(tmp_path / "missing.py")
    except OSError as e:
        pytest.skip(f"symlink is not supported: {e}")
    dist = make_record_dist(mocker, tmp_path, {}, extra_rows=[f"link.py,{_record_hash(b'x')},1"])
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_skips_editable_distributions(mocker, tmp_path):
    dist = make_record_dist(mocker, tmp_path, {"__editable__.demo.pth": b"/tmp/src\n"}, editable=True)
    (tmp_path / "__editable__.demo.pth").unlink()
    assert is_distribution_damaged(dist) is False


def test_damaged_detection_skips_distributions_without_record(mocker):
    dist = mocker.Mock()
    dist.read_text.return_value = None
    mocker.patch("pdm.installers.base.is_editable", return_value=False)
    assert is_distribution_damaged(dist) is False


def test_damaged_detection_catches_malformed_record(mocker):
    dist = mocker.Mock()
    dist.read_text.side_effect = lambda name: "bad-row\n" if name == "RECORD" else None
    mocker.patch("pdm.installers.base.is_editable", return_value=False)
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_catches_missing_cache_referral(mocker, tmp_path):
    cache_root = tmp_path / "missing-cache"
    dist = make_record_dist(mocker, tmp_path, {"demo.py": b"x\n"}, texts={"REFER_TO": str(cache_root)})
    assert is_distribution_damaged(dist) is True


def test_damaged_detection_catches_relinked_cache_symlink(mocker, tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "demo.py").write_bytes(b"x\n")
    link = tmp_path / "demo.py"
    try:
        link.symlink_to(cache_root / "demo.py")
    except OSError as e:
        pytest.skip(f"symlink is not supported: {e}")
    dist = make_record_dist(mocker, tmp_path, {}, extra_rows=["demo.py,,2"], texts={"REFER_TO": str(cache_root)})
    assert is_distribution_damaged(dist) is False
    link.unlink()
    outside = tmp_path / "other.py"
    outside.write_bytes(b"x\n")
    link.symlink_to(outside)
    assert is_distribution_damaged(dist) is True


class _FakeWorkingSet(dict):
    def __init__(self, owned_keys: set[str]):
        super().__init__()
        self._owned_keys = owned_keys

    def is_owned(self, key: str) -> bool:
        return key in self._owned_keys


def _make_compare_synchronizer(mocker, owned_keys):
    synchronizer = make_synchronizer(mocker)
    synchronizer.repair = True
    synchronizer.clean = False
    synchronizer.only_keep = False
    synchronizer.__dict__["candidates"] = {"demo": mocker.Mock(), "other": mocker.Mock()}
    damaged, healthy = mocker.Mock(), mocker.Mock()
    working_set = _FakeWorkingSet(owned_keys)
    working_set.update({"demo": damaged, "other": healthy})
    synchronizer.working_set = working_set
    mocker.patch.object(synchronizer, "_should_update", return_value=False)
    mocker.patch("pdm.installers.base.is_distribution_damaged", side_effect=lambda dist: dist is damaged)
    synchronizer.environment.project.get_locked_repository.return_value.all_candidates = []
    return synchronizer


def test_compare_repairs_only_damaged_owned_distribution(mocker):
    synchronizer = _make_compare_synchronizer(mocker, {"demo", "other"})

    assert synchronizer.compare_with_working_set() == ([], ["demo"], [])

    synchronizer.repair = False
    assert synchronizer.compare_with_working_set() == ([], [], [])


def test_compare_repair_non_owned_distribution_goes_to_add(mocker):
    synchronizer = _make_compare_synchronizer(mocker, set())

    to_add, to_update, to_remove = synchronizer.compare_with_working_set()
    assert to_add == ["demo"]
    assert to_update == []
    assert to_remove == []
