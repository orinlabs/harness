"""Load, render, and hash repo-managed agent bundles.

A bundle lives in an *agents dir* (the checkout of the private agents
repo, or any directory with the same layout):

    <agents-dir>/<name>.yaml        # RepoAgentManifest (see harness.spec)
    <agents-dir>/<name>/...         # prompt files, SOPs, sandbox files
    <agents-dir>/shared/...         # fragments shared across bundles

All paths inside a manifest resolve relative to the *agents dir* (the
manifest's parent directory), never the process cwd -- the same bundle
must render identically from production boot (cwd=/workspace/harness),
an environments-repo rollout (cwd=trial workdir), and CI.

The *bundle hash* is a sha256 over the manifest file plus every file it
references (sorted by relative path, path included in the digest). It is
the change-detection and mismatch-detection currency shared with
Bedrock: CI stamps it into the sync manifest; Bedrock stores it and can
hand it back for verification. Hashing the resolved file closure (not
the bundle folder) means an edit to a ``shared/`` fragment changes the
hash of every bundle that includes it.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness.spec import SPEC_VERSION, RepoAgentManifest

AGENTS_DIR_ENV = "HARNESS_AGENTS_DIR"

# Cheap tripwires for credentials committed into bundle files. Bundles are
# replicated into CI payloads, Bedrock's DB, and agent sandboxes -- treat
# their contents as public. This is a guardrail, not a scanner; CI may run
# a real secret scanner on top.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI-style secret key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("bearer literal", re.compile(r"bearer_literal")),
)


class BundleError(ValueError):
    """A bundle failed to load, validate, or render."""


@dataclass(frozen=True)
class LoadedBundle:
    """A validated bundle with its prompt rendered and closure hashed."""

    manifest: RepoAgentManifest
    manifest_path: Path
    agents_dir: Path
    system_prompt: str
    bundle_hash: str

    @property
    def name(self) -> str:
        return self.manifest.name

    def read_file(self, rel_path: str) -> bytes:
        return (self.agents_dir / rel_path).read_bytes()


# ---------------------------------------------------------------------------
# Directory / discovery
# ---------------------------------------------------------------------------


def resolve_agents_dir(explicit: str | Path | None = None) -> Path:
    """Precedence: explicit arg > $HARNESS_AGENTS_DIR > <cwd>/agents."""
    if explicit:
        return Path(explicit).resolve()
    if env := os.environ.get(AGENTS_DIR_ENV):
        return Path(env).resolve()
    return (Path.cwd() / "agents").resolve()


def list_bundle_names(agents_dir: Path) -> list[str]:
    """Bundle names = ``*.yaml``/``*.yml`` files that parse as manifests.

    Non-manifest agent configs (e.g. the legacy standalone schema with a
    top-level ``id``) are skipped so a mixed directory doesn't explode.
    """
    names: list[str] = []
    for path in sorted(agents_dir.glob("*.y*ml")):
        data = _read_yaml(path)
        if isinstance(data, dict) and "spec_version" in data:
            names.append(path.stem)
    return names


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_bundle(name: str, agents_dir: str | Path | None = None) -> LoadedBundle:
    """Load ``<agents-dir>/<name>.yaml``, validate, render, and hash it."""
    base = resolve_agents_dir(agents_dir)
    manifest_path = _find_manifest(base, name)
    data = _read_yaml(manifest_path)
    if not isinstance(data, dict):
        raise BundleError(f"{manifest_path}: expected a mapping at the top level")

    try:
        manifest = RepoAgentManifest.model_validate(data)
    except Exception as e:  # pydantic.ValidationError, but avoid the import
        raise BundleError(f"{manifest_path}: {e}") from e

    if manifest.name != name:
        raise BundleError(
            f"{manifest_path}: manifest name {manifest.name!r} must match the file stem {name!r}"
        )

    referenced = _referenced_files(manifest)
    missing = [rel for rel in referenced if not (base / rel).is_file()]
    if missing:
        raise BundleError(f"{manifest_path}: referenced file(s) not found: {', '.join(missing)}")

    # Files consumed as text (prompt sources and document targets) must
    # decode as UTF-8. Checking here keeps `validate-agents` (the CI gate)
    # in agreement with rendering, which would otherwise crash with a raw
    # UnicodeDecodeError at sync time.
    for rel in _text_files(manifest):
        try:
            (base / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise BundleError(
                f"{manifest_path}: {rel} is not valid UTF-8 "
                "(prompt and document files must be UTF-8 text)"
            ) from e

    system_prompt = _render_system_prompt(manifest, base)
    bundle_hash = _compute_bundle_hash(manifest_path, base, referenced)
    return LoadedBundle(
        manifest=manifest,
        manifest_path=manifest_path,
        agents_dir=base,
        system_prompt=system_prompt,
        bundle_hash=bundle_hash,
    )


def scan_for_secrets(bundle: LoadedBundle) -> list[str]:
    """Return human-readable findings for secret-looking content.

    Scans the manifest and every referenced text file. Binary files
    (undecodable as UTF-8) are skipped -- they can't carry a copy-pasted
    token in any of the shapes we grep for.
    """
    findings: list[str] = []
    paths = [bundle.manifest_path] + [
        bundle.agents_dir / rel for rel in _referenced_files(bundle.manifest)
    ]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path}: looks like it contains a {label}")
    return findings


# ---------------------------------------------------------------------------
# Rendering for the sync manifest (consumed by Bedrock's repo-agents/sync)
# ---------------------------------------------------------------------------


def render_bundle_payload(bundle: LoadedBundle) -> dict:
    """One agent entry of the sync manifest.

    Wire shape (versioned by ``spec_version``):

        {
          "name", "spec_version", "bundle_hash",
          "config": {... repo-owned Agent fields, rendered prompt ...},
          "documents": [{"title", "kind", "content"}],
          "sandbox_files": {"<dest>": "<base64>"}
        }
    """
    m = bundle.manifest
    documents: list[dict] = []
    sandbox_files: dict[str, str] = {}
    for f in m.files:
        raw = bundle.read_file(f.path)
        if f.target == "document":
            documents.append(
                {
                    "title": f.title or Path(f.path).stem,
                    "kind": f.kind,
                    "content": raw.decode("utf-8"),
                }
            )
        else:
            dest = f.dest or f.path
            sandbox_files[dest] = base64.b64encode(raw).decode("ascii")

    config = {
        "display_name": m.display_name or m.name,
        "model": m.model,
        "reasoning_effort": m.reasoning_effort,
        "max_tokens": m.max_tokens,
        "max_turns": m.max_turns,
        "max_sleep_hours": m.max_sleep_hours,
        "timezone": m.timezone,
        "system_prompt": bundle.system_prompt,
        "adapters": list(m.adapters),
        "expected_tools": list(m.expected_tools),
        "feature_flags": dict(m.feature_flags),
        "memory": m.memory.model_dump(),
        "adopt_agent_id": m.adopt_agent_id,
        "previous_names": list(m.previous_names),
    }
    return {
        "name": m.name,
        "spec_version": m.spec_version,
        "bundle_hash": bundle.bundle_hash,
        "config": config,
        "documents": documents,
        "sandbox_files": sandbox_files,
    }


def render_sync_manifest(
    agents_dir: str | Path | None = None,
    *,
    harness_git_sha: str | None = None,
) -> dict:
    """The full payload CI POSTs to Bedrock's repo-agents/sync endpoint."""
    base = resolve_agents_dir(agents_dir)
    agents = []
    for name in list_bundle_names(base):
        bundle = load_bundle(name, base)
        agents.append(render_bundle_payload(bundle))
    payload: dict = {"spec_version": SPEC_VERSION, "agents": agents}
    if harness_git_sha:
        payload["harness_git_sha"] = harness_git_sha
    return payload


# ---------------------------------------------------------------------------
# Merge: bundle + overlay -> AgentConfig dict
# ---------------------------------------------------------------------------


def expand_extends(data: dict, *, agents_dir: str | Path | None = None) -> dict:
    """Expand an ``extends: <bundle-name>`` config dict into a flat one.

    Used by the config loader when a local agent config (typically the
    environments repo's generated trial JSON) inherits from a bundle so
    evals exercise the exact production prompt/config. Overlay semantics
    mirror Bedrock's portal addendum: the deployment supplies identity,
    tools, and additions; the bundle supplies the agent's substance.

        {
          "id": "<trial id>",              # required; runtime identity
          "extends": "bidlevel",
          "tools": [...],                  # env/world tools (overlay-owned)
          "model": "...",                  # optional; overrides the bundle pin
          "system_prompt_prefix": "...",   # e.g. the rollout contract preamble
          "system_prompt_suffix": "...",   # e.g. the scenario task prompt
          "feature_flags": {...},          # overlaid onto the bundle's flags
        }

    Bundle resolution: explicit ``agents_dir`` arg > ``$HARNESS_AGENTS_DIR``
    > ``<cwd>/agents``.
    """
    if "extends" not in data:
        return data
    if "system_prompt" in data:
        raise BundleError(
            "config with `extends` must not also set `system_prompt`; "
            "use system_prompt_prefix / system_prompt_suffix"
        )
    if not data.get("id"):
        raise BundleError("config with `extends` must set `id` (the runtime agent id)")

    bundle = load_bundle(str(data["extends"]), agents_dir)
    m = bundle.manifest

    parts = [
        str(data.get("system_prompt_prefix") or "").strip(),
        bundle.system_prompt,
        str(data.get("system_prompt_suffix") or "").strip(),
    ]
    system_prompt = "\n\n".join(p for p in parts if p)

    merged: dict = {
        "id": data["id"],
        "model": data.get("model") or m.model,
        "system_prompt": system_prompt,
        "reasoning_effort": data.get("reasoning_effort") or m.reasoning_effort,
        "max_tokens": data.get("max_tokens") or m.max_tokens,
        "timezone": data.get("timezone") or m.timezone,
        "feature_flags": {**m.feature_flags, **(data.get("feature_flags") or {})},
        "memory": m.memory.model_dump(),
        "tools": data.get("tools", []),
    }
    return merged


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_manifest(agents_dir: Path, name: str) -> Path:
    for ext in (".yaml", ".yml"):
        p = agents_dir / f"{name}{ext}"
        if p.is_file():
            return p
    raise BundleError(f"no bundle manifest for {name!r} in {agents_dir}")


def _read_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise BundleError(f"{path}: invalid YAML: {e}") from e


def _referenced_files(manifest: RepoAgentManifest) -> list[str]:
    """All bundle-relative file paths the manifest references, sorted."""
    refs: set[str] = set()
    if manifest.system_prompt_file:
        refs.add(manifest.system_prompt_file)
    refs.update(manifest.prompt_fragments)
    refs.update(f.path for f in manifest.files)
    return sorted(refs)


def _text_files(manifest: RepoAgentManifest) -> list[str]:
    """Referenced files that get decoded as UTF-8 text, sorted."""
    refs: set[str] = set(manifest.prompt_fragments)
    if manifest.system_prompt_file:
        refs.add(manifest.system_prompt_file)
    refs.update(f.path for f in manifest.files if f.target == "document")
    return sorted(refs)


def _render_system_prompt(manifest: RepoAgentManifest, base: Path) -> str:
    if manifest.system_prompt is not None:
        parts = [manifest.system_prompt]
    else:
        assert manifest.system_prompt_file is not None
        parts = [(base / manifest.system_prompt_file).read_text(encoding="utf-8")]
    for rel in manifest.prompt_fragments:
        parts.append((base / rel).read_text(encoding="utf-8"))
    return "\n\n".join(p.strip() for p in parts if p.strip())


def _compute_bundle_hash(manifest_path: Path, base: Path, referenced: list[str]) -> str:
    """sha256 over the manifest + the resolved file closure.

    Relative paths are mixed into the digest so a rename changes the
    hash even when contents don't. Each entry contributes
    ``<rel>\\x00<sha256(content) hex>\\n``: hashing a fixed-length digest
    of the content (rather than the raw bytes, which may themselves
    contain NULs) keeps the encoding injective, so distinct closures
    can't collide by crafting content that mimics entry framing.
    """
    h = hashlib.sha256()
    entries = [(manifest_path.name, manifest_path)] + [(rel, base / rel) for rel in referenced]
    for rel, path in entries:
        content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        h.update(f"{rel}\x00{content_digest}\n".encode())
    return f"sha256:{h.hexdigest()}"
