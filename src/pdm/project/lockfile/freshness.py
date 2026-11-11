from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

from dep_logic.markers import AnyMarker, BaseMarker, MarkerUnion

from pdm.exceptions import PdmException, RequirementError
from pdm.models.markers import exclude_multi
from pdm.models.requirements import (
    ALLOW_ANY,
    FileRequirement,
    NamedRequirement,
    Requirement,
    VcsRequirement,
    parse_line,
    parse_requirement,
    strip_extras,
)
from pdm.utils import cd, get_file_hash, get_requirement_from_override, normalize_name, url_without_fragments

if TYPE_CHECKING:
    from pathlib import Path

    from pdm.models.markers import EnvSpec
    from pdm.models.repositories.lock import LockedRepository, Package
    from pdm.project import Project


_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_OPAQUE_REQUIREMENT_PREFIX = "url-sha256:"
_DYNAMIC_LOCK_FIELDS = frozenset({"dependencies", "optional-dependencies", "requires-python", "version"})


def _to_builtin(value: Any, *, allow_temporal: bool = False) -> Any:
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if isinstance(value, Mapping):
        return {key: _to_builtin(item, allow_temporal=allow_temporal) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item, allow_temporal=allow_temporal) for item in value]
    if allow_temporal and isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    raise TypeError(f"Unsupported lock input type: {type(value).__name__}")


def _fingerprint(value: str, prefix: str = "sha256:") -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _requirement_input(requirement: Requirement) -> str:
    line = requirement.as_line()
    if isinstance(requirement, NamedRequirement) and not _URL_RE.search(line):
        return dataclasses.replace(requirement, specifier=ALLOW_ANY).as_line()
    if isinstance(requirement, FileRequirement) and not requirement.url and requirement.path is not None:
        line = json.dumps(
            {
                "editable": requirement.editable,
                "extras": sorted(requirement.extras or []),
                "marker": str(requirement.marker or ""),
                "name": requirement.project_name or "",
                "path": requirement.str_path,
                "subdirectory": requirement.subdirectory or "",
            },
            sort_keys=True,
        )
    return _fingerprint(line, _OPAQUE_REQUIREMENT_PREFIX)


def _sanitize_strings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_strings(item) for item in value]
    if isinstance(value, str) and _URL_RE.search(value):
        return _fingerprint(value)
    return value


def _normalize_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        item.pop("username", None)
        item.pop("password", None)
        if url := item.get("url"):
            item["url"] = _fingerprint(str(url))
        for key in ("include_packages", "exclude_packages"):
            if key in item:
                item[key] = sorted(item[key])
        result.append(_sanitize_strings(item))
    return result


def _project_lock_inputs(
    project: Project,
    include_package: bool = False,
    managed_paths: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    dependencies = project._resolve_dependencies()
    resolution = _to_builtin(project.pyproject.resolution, allow_temporal=True)
    resolution.pop("lock_inputs", None)
    with cd(project.root):
        groups = {
            group: sorted(_requirement_input(requirement) for requirement in requirements)
            for group, requirements in sorted(dependencies.items())
        }

    result: dict[str, Any] = {
        "groups": groups,
        "requires-python": str(project.python_requires),
        "resolution": _sanitize_strings(resolution),
        "sources": _normalize_sources(_to_builtin(project.pyproject.settings.get("source", []), allow_temporal=True)),
    }
    if include_package:
        metadata = project.pyproject.metadata
        dynamic = sorted(metadata.get("dynamic", []))
        result["package"] = {
            "build-system": _sanitize_strings(_to_builtin(project.pyproject.build_system, allow_temporal=True)),
            "distribution": project.is_distribution,
            "dynamic": dynamic,
            "name": metadata.get("name", ""),
            "version": metadata.get("version", ""),
        }
        if _DYNAMIC_LOCK_FIELDS.intersection(dynamic):
            result["volatile"] = True

    local_inputs: dict[str, Any] = {}
    for requirements in dependencies.values():
        for requirement in requirements:
            if not isinstance(requirement, FileRequirement) or requirement.absolute_path is None:
                continue
            path = requirement.absolute_path.resolve()
            if path in managed_paths:
                continue
            key = _requirement_input(requirement)
            if path.is_file():
                local_inputs[key] = {"file-sha256": get_file_hash(path)}
            elif path.is_dir():
                local_project = project.core.create_project(path)
                if local_project.pyproject.is_valid:
                    local_inputs[key] = _project_lock_inputs(
                        local_project,
                        include_package=True,
                        managed_paths=managed_paths | {path},
                    )
                else:
                    local_inputs[key] = {"volatile": True}
    if local_inputs:
        result["local"] = {key: local_inputs[key] for key in sorted(local_inputs)}
    return result


def build_lock_inputs(project: Project) -> dict[str, Any]:
    """Build the canonical project inputs that determine a lock resolution."""
    workspace_project = project.workspace_project or project
    root = workspace_project.root.resolve()
    members = (
        sorted(workspace_project.iter_members(), key=lambda path: path.relative_to(root).as_posix())
        if workspace_project.is_workspace_root
        else []
    )
    managed_paths = frozenset({root, *(member.resolve() for member in members)})
    result: dict[str, Any] = {
        "project": _project_lock_inputs(workspace_project, managed_paths=managed_paths),
    }
    if members:
        result["workspace"] = {
            member.relative_to(root).as_posix(): _project_lock_inputs(
                workspace_project.core.create_project(member),
                include_package=True,
                managed_paths=managed_paths,
            )
            for member in members
        }
    return result


def _contains_volatile(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("volatile") is True or any(_contains_volatile(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_volatile(item) for item in value)
    return False


def _requirements_match(
    current_lines: Sequence[str],
    locked_lines: Sequence[str],
) -> bool:
    if len(current_lines) != len(locked_lines):
        return False
    if not all(isinstance(line, str) for line in (*current_lines, *locked_lines)):
        return False
    return sorted(current_lines) == sorted(locked_lines)


def _project_inputs_match(
    current: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> bool:
    current_metadata = {key: value for key, value in current.items() if key != "groups"}
    locked_metadata = {key: value for key, value in locked.items() if key != "groups"}
    if current_metadata != locked_metadata:
        return False

    current_groups = current.get("groups")
    locked_groups = locked.get("groups")
    if not isinstance(current_groups, Mapping) or not isinstance(locked_groups, Mapping):
        return False
    if current_groups.keys() != locked_groups.keys():
        return False
    return all(
        isinstance(current_groups[group], Sequence)
        and not isinstance(current_groups[group], str)
        and isinstance(locked_groups[group], Sequence)
        and not isinstance(locked_groups[group], str)
        and _requirements_match(current_groups[group], locked_groups[group])
        for group in current_groups
    )


def _project_overrides(project: Project) -> dict[str, Requirement] | None:
    result: dict[str, Requirement] = {}
    try:
        with cd(project.root):
            for name, value in project.pyproject.resolution.get("overrides", {}).items():
                requirement = parse_requirement(get_requirement_from_override(normalize_name(name), value))
                result[requirement.identify()] = requirement
    except (RequirementError, TypeError, ValueError):
        return None
    return result


def _effective_requirement(requirement: Requirement, overrides: Mapping[str, Requirement]) -> Requirement:
    effective = overrides.get(requirement.identify())
    if effective is None and requirement.key is not None:
        effective = overrides.get(requirement.key)
    return requirement if effective is None else effective


def _requirement_is_excluded(requirement: Requirement, excludes: frozenset[str]) -> bool:
    return requirement.identify() in excludes or (requirement.key is not None and requirement.key in excludes)


def _requirement_marker(requirement: Requirement) -> BaseMarker:
    return requirement.marker.inner if requirement.marker is not None else AnyMarker()


def _package_marker(package: Package) -> BaseMarker:
    marker = package.candidate.req.marker
    return marker.inner if marker is not None else AnyMarker()


def _applicable_regions(requirement: Requirement, repository: LockedRepository) -> list[BaseMarker] | None:
    if not repository.targets:
        return None
    requirement_marker = _requirement_marker(requirement)
    regions: list[BaseMarker] = []
    for target in repository.targets:
        region = requirement_marker & target.markers_with_python().inner
        if not region.is_empty():
            regions.append(region)
    return regions


def _package_matches_requirement_context(
    package: Package,
    requirement: Requirement,
    project: Project,
    group: str,
) -> bool:
    candidate = package.candidate
    if requirement.key is not None and normalize_name(candidate.name or "") != requirement.key:
        return False
    if candidate.req.groups and group not in candidate.req.groups:
        return False
    extras, dependency_groups = project.split_extras_groups([group])
    return package.marker.evaluate({"extras": set(extras), "dependency_groups": set(dependency_groups)})


def _context_packages(
    requirement: Requirement,
    project: Project,
    group: str,
    repository: LockedRepository,
    regions: Sequence[BaseMarker],
) -> list[Package]:
    return [
        package
        for package in repository.packages.values()
        if _package_matches_requirement_context(package, requirement, project, group)
        and any(not (region & _package_marker(package)).is_empty() for region in regions)
    ]


def _packages_cover_regions(packages: Sequence[Package], regions: Sequence[BaseMarker]) -> bool:
    if not packages:
        return False
    coverage = MarkerUnion.of(*(_package_marker(package) for package in packages))
    return all((region & coverage) == region for region in regions)


def _file_source_identity(requirement: FileRequirement, project: Project) -> tuple[Any, ...]:
    common = (requirement.editable, requirement.subdirectory or "")
    if requirement.absolute_path is not None:
        return ("path", requirement.absolute_path.resolve(), *common)
    if isinstance(requirement, VcsRequirement):
        repo = project.backend.expand_line(url_without_fragments(requirement.repo))
        return ("vcs", requirement.vcs, repo, requirement.ref or "", *common)
    url = project.backend.expand_line(url_without_fragments(requirement.get_full_url()))
    return ("url", url, *common)


def _requirement_matches(
    requirement: Requirement,
    effective: Requirement,
    project: Project,
    group: str,
    repository: LockedRepository,
) -> bool:
    regions = _applicable_regions(requirement, repository)
    if regions is None:
        return False
    if not regions:
        return True
    packages = _context_packages(requirement, project, group, repository, regions)
    if isinstance(effective, NamedRequirement):
        if not all(
            package.candidate.version is not None
            and effective.specifier.contains(package.candidate.version, prereleases=True)
            for package in packages
        ):
            return False
    elif isinstance(effective, FileRequirement):
        expected_source = _file_source_identity(effective, project)
        packages = [
            package
            for package in packages
            if isinstance(package.candidate.req, FileRequirement)
            and _file_source_identity(package.candidate.req, project) == expected_source
        ]
    else:  # pragma: no cover
        return False
    return _packages_cover_regions(packages, regions)


def _project_specifiers_match(
    project: Project,
    repository: LockedRepository,
    overrides: Mapping[str, Requirement],
    excludes: frozenset[str],
    managed_paths: frozenset[Path],
    groups: Sequence[str],
) -> bool:
    return _project_specifier_failures(project, repository, overrides, excludes, managed_paths, groups) == []


def _project_specifier_failures(
    project: Project,
    repository: LockedRepository,
    overrides: Mapping[str, Requirement],
    excludes: frozenset[str],
    managed_paths: frozenset[Path],
    groups: Sequence[str],
) -> list[Requirement] | None:
    """Return the direct requirements whose locked candidates don't satisfy them.

    Returns ``None`` when the inputs cannot be validated reliably and the
    caller should fall back to a full re-resolution.
    """
    failures: list[Requirement] = []
    dependencies = project._resolve_dependencies(list(groups))
    for group, requirements in dependencies.items():
        for requirement in requirements:
            if _requirement_is_excluded(requirement, excludes):
                continue
            effective = _effective_requirement(requirement, overrides)
            if not _requirement_matches(requirement, effective, project, group, repository):
                failures.append(requirement)
            if not isinstance(effective, FileRequirement) or effective.absolute_path is None:
                continue
            path = effective.absolute_path.resolve()
            if path in managed_paths or not path.is_dir():
                continue
            local_project = project.core.create_project(path)
            if not local_project.pyproject.is_valid:
                return None
            nested_failures = _project_specifier_failures(
                local_project,
                repository,
                overrides,
                excludes,
                managed_paths | {path},
                ["default"],
            )
            if nested_failures is None:
                return None
            failures.extend(nested_failures)
    return failures


def _workspace_specifier_failures(project: Project, repository: LockedRepository) -> list[Requirement] | None:
    root = project.root.resolve()
    members = (
        sorted(project.iter_members(), key=lambda path: path.relative_to(root).as_posix())
        if project.is_workspace_root
        else []
    )
    managed_paths = frozenset({root, *(member.resolve() for member in members)})
    overrides = _project_overrides(project)
    if overrides is None:
        return None
    excludes = frozenset(normalize_name(name) for name in project.pyproject.resolution.get("excludes", []))
    failures = _project_specifier_failures(
        project,
        repository,
        overrides,
        excludes,
        managed_paths,
        project.lockfile.groups or ["default"],
    )
    if failures is None:
        return None
    for member in members:
        member_failures = _project_specifier_failures(
            project.core.create_project(member),
            repository,
            overrides,
            excludes,
            managed_paths,
            ["default"],
        )
        if member_failures is None:
            return None
        failures.extend(member_failures)
    return failures


def _all_specifiers_match(project: Project, repository: LockedRepository) -> bool:
    failures = _workspace_specifier_failures(project, repository)
    return failures is not None and not failures


def lock_inputs_match(project: Project, locked_inputs: object) -> bool:
    """Return whether the current project is satisfied by the recorded lock inputs."""
    try:
        locked = _to_builtin(locked_inputs)
    except (TypeError, ValueError):
        return False
    if not isinstance(locked, Mapping):
        return False
    current = build_lock_inputs(project)
    if _contains_volatile(current) or _contains_volatile(locked):
        return False

    workspace_project = project.workspace_project or project
    if current != locked:
        current_project = current.get("project")
        locked_project = locked.get("project")
        if not isinstance(current_project, Mapping) or not isinstance(locked_project, Mapping):
            return False
        if not _project_inputs_match(current_project, locked_project):
            return False

        current_workspace = current.get("workspace", {})
        locked_workspace = locked.get("workspace", {})
        if not isinstance(current_workspace, Mapping) or not isinstance(locked_workspace, Mapping):
            return False
        if current_workspace.keys() != locked_workspace.keys():
            return False
        if not all(
            isinstance(current_workspace[path], Mapping)
            and isinstance(locked_workspace[path], Mapping)
            and _project_inputs_match(current_workspace[path], locked_workspace[path])
            for path in current_workspace
        ):
            return False

    try:
        return _all_specifiers_match(workspace_project, workspace_project.get_locked_repository())
    except (PdmException, TypeError, ValueError):
        return False


def _diff_group_inputs(current_groups: Any, locked_groups: Any) -> list[str] | None:
    """Return the groups whose requirement inputs changed.

    Returns ``None`` when the group layout itself changed (a group was added or
    removed), which requires a full re-resolution.
    """
    if not isinstance(current_groups, Mapping) or not isinstance(locked_groups, Mapping):
        return None
    if current_groups.keys() != locked_groups.keys():
        return None
    return [group for group in current_groups if not _requirements_match(current_groups[group], locked_groups[group])]


def _diff_project_inputs(current_project: Any, locked_project: Any) -> list[str] | None:
    """Diff one project's lock inputs, returning the changed groups.

    ``None`` means the difference cannot be handled incrementally.
    """
    if not isinstance(current_project, Mapping) or not isinstance(locked_project, Mapping):
        return None
    changed_groups = _diff_group_inputs(current_project.get("groups"), locked_project.get("groups"))
    if changed_groups is None:
        return None
    current_metadata = {key: value for key, value in current_project.items() if key != "groups"}
    locked_metadata = {key: value for key, value in locked_project.items() if key != "groups"}
    if current_metadata != locked_metadata:
        return None
    return changed_groups


def diff_lock_inputs(current: Any, locked: Any) -> dict[str, list[str]] | None:
    """Compare canonical lock inputs and report changed groups per project.

    The result maps a project key (``"project"`` for the workspace root or a
    workspace member path) to the groups whose requirement fingerprints
    changed. ``None`` means the difference cannot be handled incrementally and
    a full re-resolution is required: unknown input shapes, added or removed
    workspace members/groups, or changes to non-group metadata such as
    ``requires-python``, resolution settings, sources or local artifacts.
    """
    try:
        current = _to_builtin(current)
        locked = _to_builtin(locked)
    except (TypeError, ValueError):
        return None
    if not isinstance(current, Mapping) or not isinstance(locked, Mapping):
        return None
    if current.keys() != locked.keys():
        return None
    changed: dict[str, list[str]] = {}
    root_diff = _diff_project_inputs(current.get("project"), locked.get("project"))
    if root_diff is None:
        return None
    if root_diff:
        changed["project"] = root_diff

    current_workspace = current.get("workspace") or {}
    locked_workspace = locked.get("workspace") or {}
    if not isinstance(current_workspace, Mapping) or not isinstance(locked_workspace, Mapping):
        return None
    if current_workspace.keys() != locked_workspace.keys():
        return None
    for member_key, current_member in current_workspace.items():
        member_diff = _diff_project_inputs(current_member, locked_workspace[member_key])
        if member_diff is None:
            return None
        if member_diff:
            changed[member_key] = member_diff
    return changed


def _expand_dependency_closure(repository: LockedRepository, roots: AbstractSet[str]) -> set[str]:
    """Expand direct requirement names to the full transitive closure in the lock."""
    seen = set(roots)
    pending = deque(seen)
    while pending:
        name = pending.popleft()
        for key, entry in repository.packages.items():
            if key[0] != name or not entry.dependencies:
                continue
            for line in entry.dependencies:
                dependency_name = strip_extras(parse_line(line).identify())[0]
                if dependency_name not in seen:
                    seen.add(dependency_name)
                    pending.append(dependency_name)
    return seen


def _marker_matches_target(marker: Any, target: EnvSpec) -> bool:
    bare_marker = exclude_multi(marker, "extras", "dependency_groups")
    return bare_marker.matches(target)


def build_incremental_lock_plan(
    project: Project,
    repository: LockedRepository,
    current_inputs: Mapping[str, Any] | None = None,
    locked_inputs: Any = None,
    targets: Sequence[EnvSpec] | None = None,
) -> dict[str, Any] | None:
    """Build a plan for re-resolving only the affected dependency closure.

    Returns a mapping with the changed groups, the tracked package names (the
    transitive closure that may move during resolution) and the lock targets
    affected by the changed requirements. Returns ``None`` when incremental
    locking is not applicable and the caller should fall back to a full
    re-resolution.
    """
    if current_inputs is None:
        current_inputs = build_lock_inputs(project)
    if locked_inputs is None:
        locked_inputs = project.lockfile.lock_inputs
    if _contains_volatile(current_inputs) or _contains_volatile(locked_inputs):
        return None
    changed_groups = diff_lock_inputs(current_inputs, locked_inputs)
    if changed_groups is None:
        return None

    workspace_project = project.workspace_project or project
    root = workspace_project.root.resolve()
    roots: set[str] = set()
    changed_requirements: list[Requirement] = []
    for project_key, groups in changed_groups.items():
        if project_key == "project":
            sub_project = workspace_project
        else:
            member_path = root / project_key
            if not member_path.is_dir():
                return None
            sub_project = workspace_project.core.create_project(member_path)
        dependencies = sub_project._resolve_dependencies(list(groups))
        for group in groups:
            group_requirements = dependencies.get(group, [])
            if not group_requirements:
                # A group became empty: without requirement markers the affected
                # targets cannot be determined, fall back to a full re-resolution.
                return None
            for requirement in group_requirements:
                if requirement.key is None:
                    return None
                roots.add(normalize_name(requirement.key))
                changed_requirements.append(requirement)
        if project_key != "project" and sub_project.name:
            roots.add(normalize_name(sub_project.name))

    # Specifier-only edits don't change the requirement fingerprints, so detect
    # the locked candidates that no longer satisfy the current requirements.
    specifier_failures = _workspace_specifier_failures(workspace_project, repository)
    if specifier_failures is None:
        return None
    for requirement in specifier_failures:
        if requirement.key is None:
            return None
        roots.add(normalize_name(requirement.key))
        changed_requirements.append(requirement)

    tracked_names = _expand_dependency_closure(repository, roots) if roots else set()
    candidate_targets = list(repository.targets if targets is None else targets)
    affected_targets = [
        target
        for target in candidate_targets
        if any(
            requirement.marker is None or _marker_matches_target(requirement.marker, target)
            for requirement in changed_requirements
        )
    ]
    return {
        "changed_groups": changed_groups,
        "tracked_names": tracked_names,
        "affected_targets": affected_targets,
    }
