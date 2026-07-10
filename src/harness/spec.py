"""Repo-managed agent bundle spec.

A *bundle* is the git-authored definition of a production agent: a
``<name>.yaml`` manifest plus an optional ``<name>/`` folder holding the
system prompt, SOP/prompt fragments, and files to ship. Bundles live in
the (private) agents repo, NOT in this repo -- the harness only owns the
schema and the validation/rendering tooling (``harness validate-agents``
and ``harness render-agent``).

Contract notes:

* ``SPEC_VERSION`` is the manifest schema version this checkout
  understands. The agents-repo CI stamps it into the sync manifest so
  Bedrock can refuse payloads newer than it supports.
* ``harness-spec.json`` at the repo root is *generated* from
  ``RepoAgentManifest.model_json_schema()``. A unit test asserts the
  committed file matches the models, so external consumers (Bedrock, the
  portal, the agents repo) get a machine-readable schema without a
  second hand-maintained source of truth. Regenerate with:

      uv run python -m harness.spec > harness-spec.json

* Manifests deliberately have no ``id``: the runtime agent id is a
  Bedrock UUID (production) or a trial id (evals), injected by whoever
  instantiates the bundle. ``name`` identifies the bundle itself.
* Manifests must never carry secrets. Adapter credentials live in
  Bedrock's per-agent adapter configs; tool endpoints/auth are deployment
  facts stamped by Bedrock at run time.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = 1

# Bundle names double as file/dir names and Bedrock's ``repo_agent_name``
# key; keep them boring.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# The memory system named by ``MemorySpec.system`` today. New backends
# register here so old manifests fail loudly on a checkout that predates
# the backend they ask for.
KNOWN_MEMORY_SYSTEMS = ("tiered_sqlite",)

DEFAULT_SUMMARIZER_MODEL = "openai/gpt-5-nano"


def _require_bundle_relative(value: str, field: str) -> None:
    """Reject absolute paths and ``..`` segments.

    Every path in a manifest resolves against the agents dir and is then
    read, hashed, and replicated into sync payloads -- an escaping path
    would exfiltrate arbitrary host files into the rendered prompt.
    """
    if value.startswith("/") or ".." in value.split("/"):
        raise ValueError(f"{field} must be a relative path inside the bundle: {value!r}")


class MemorySpec(BaseModel):
    """Per-agent memory configuration.

    Omitting the block entirely is always safe: every field's null/absent
    state means "whatever the running harness decides". That is the
    general rule for this model -- memory tunables only become spec
    fields when their absence has a well-defined "harness decides"
    meaning. Otherwise every synced config would bake in whatever the
    default was on the day it was rendered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: Literal["tiered_sqlite"] = "tiered_sqlite"
    # Summarization runs on a cheap model, never the agent's own model
    # (see Harness.__init__ for the cost rationale).
    summarizer_model: str | None = Field(
        default=None,
        description=(
            "Model for memory summarization. null/omitted means 'whatever the "
            f"running harness defaults to' (currently {DEFAULT_SUMMARIZER_MODEL!r}, "
            "resolved by the memory factory at construction time); a set value "
            "pins it. The default is deliberately NOT baked in here: if it were, "
            "every rendered config and bundle hash would silently pin today's "
            "default, and a fleet-wide summarizer upgrade would require editing "
            "every bundle yaml."
        ),
    )


class BundleFile(BaseModel):
    """A file shipped with the bundle.

    ``target`` picks the delivery mechanism:

    * ``document`` -- delivered as an agent document via the control
      plane (kind=skill by default), upserted by title on sync. Readable
      via the agent's ``get_document`` tools and visible in the portal.
    * ``sandbox`` -- shipped as a plain file into the agent's sandbox,
      applied by the control plane before the agent wakes. ``dest`` is the
      sandbox-relative destination path (defaults to the bundle-relative
      source path).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(description="Source path relative to the manifest's directory.")
    target: Literal["document", "sandbox"]
    # document-target options
    title: str | None = Field(
        default=None,
        description="Document title (document target only). Defaults to the file stem.",
    )
    kind: Literal["note", "skill"] = "skill"
    # sandbox-target options
    dest: str | None = Field(
        default=None,
        description="Destination path relative to the sandbox working dir (sandbox target only).",
    )

    @model_validator(mode="after")
    def _check_target_fields(self) -> BundleFile:
        if self.target == "document" and self.dest is not None:
            raise ValueError("files[].dest is only valid with target: sandbox")
        if self.target == "sandbox" and self.title is not None:
            raise ValueError("files[].title is only valid with target: document")
        _require_bundle_relative(self.path, "files[].path")
        if self.dest is not None:
            _require_bundle_relative(self.dest, "files[].dest")
        return self


class RepoAgentManifest(BaseModel):
    """Schema for ``agents/<name>.yaml`` in the agents repo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: int = Field(ge=1)
    name: str
    display_name: str | None = Field(
        default=None,
        description="Human-facing agent name in Bedrock. Defaults to `name`.",
    )

    # --- model / loop settings (repo-owned in production) ---------------
    model: str
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_turns: int | None = Field(default=None, gt=0)
    max_sleep_hours: int | None = Field(default=None, gt=0)
    timezone: str | None = None

    # --- prompt ----------------------------------------------------------
    system_prompt: str | None = Field(
        default=None,
        description="Literal prompt. Mutually exclusive with system_prompt_file.",
    )
    system_prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt file, relative to the manifest's directory.",
    )
    prompt_fragments: list[str] = Field(
        default_factory=list,
        description="Extra prompt files appended after the system prompt, in order.",
    )

    # --- shipped files ----------------------------------------------------
    files: list[BundleFile] = Field(default_factory=list)

    # --- bedrock-owned surfaces the bundle declares ------------------------
    adapters: list[str] = Field(
        default_factory=list,
        description="Adapter names (presence only; credentials stay in Bedrock).",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tool names the prompt assumes exist. Bedrock warns when unsatisfied.",
    )
    feature_flags: dict[str, str] = Field(default_factory=dict)
    memory: MemorySpec = Field(default_factory=MemorySpec)

    # --- sync bookkeeping ---------------------------------------------------
    adopt_agent_id: str | None = Field(
        default=None,
        description=(
            "Existing Bedrock agent UUID to adopt in place on first sync. "
            "Honored only while that agent is not yet repo-managed."
        ),
    )
    previous_names: list[str] = Field(
        default_factory=list,
        description="Former bundle names, so a rename is not treated as delete+create.",
    )

    @model_validator(mode="after")
    def _check(self) -> RepoAgentManifest:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"name must match {_NAME_RE.pattern} (lowercase, digits, hyphens): {self.name!r}"
            )
        has_literal = self.system_prompt is not None
        has_file = self.system_prompt_file is not None
        if has_literal == has_file:
            raise ValueError("exactly one of system_prompt / system_prompt_file is required")
        if self.system_prompt_file is not None:
            _require_bundle_relative(self.system_prompt_file, "system_prompt_file")
        for fragment in self.prompt_fragments:
            _require_bundle_relative(fragment, "prompt_fragments[]")
        if self.spec_version > SPEC_VERSION:
            raise ValueError(
                f"manifest spec_version={self.spec_version} is newer than this "
                f"harness checkout supports (SPEC_VERSION={SPEC_VERSION})"
            )
        return self


def spec_json_schema() -> dict:
    """The JSON Schema committed as ``harness-spec.json``."""
    schema = RepoAgentManifest.model_json_schema()
    schema["$comment"] = (
        f"Generated from harness.spec (SPEC_VERSION={SPEC_VERSION}). "
        "Do not edit by hand: uv run python -m harness.spec > harness-spec.json"
    )
    return schema


if __name__ == "__main__":
    import json
    import sys

    json.dump(spec_json_schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
