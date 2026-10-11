"""Execution policy and preserved metric tolerance profiles."""

import re
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from schemas.claim import Contract, FiniteNumber, NonEmpty

# Existing plan generation used .02 for MRR; the execution judge used .01.
# Keep both historical contracts explicit. V2 evidence uses alignment.
TOLERANCES = {
    "legacy_plan": {"mr": 30.0, "rate": 0.02, "percentage": 2.0, "absolute": 0.02, "relative": 0.05},
    "alignment": {"mr": 30.0, "mrr": 0.01, "rate": 0.02, "percentage": 2.0, "absolute": 0.05},
}


def metric_tolerance(metric: str, expected: Any = 0, *, profile="alignment", delta=None) -> float:
    values = TOLERANCES[profile]
    key = str(metric or "").strip().lower()
    try:
        magnitude = abs(float(expected))
    except (ValueError, TypeError):
        magnitude = 0.0
    if key == "mr":
        return values["mr"]
    if key == "mrr" and profile == "alignment":
        return values["mrr"]
    rates = {"mrr", "accuracy", "acc", "f1", "precision", "recall", "auc"}
    if profile == "alignment":
        rates = (rates - {"acc"}) | {"map", "ndcg"}
    if key.startswith("hits@") or key in rates:
        return values["percentage"] if profile == "legacy_plan" and magnitude > 1 else values["rate"]
    if key in {"bleu", "rouge-l", "rouge-1", "rouge-2"}:
        # Preserve the legacy alignment convention, including its delta-based scale.
        scale = magnitude if profile == "legacy_plan" else abs(float(delta or 0))
        return values["percentage"] if scale > 1 else values["rate"]
    if profile == "legacy_plan":
        return max(values["absolute"], magnitude * values["relative"])
    return values["absolute"]


class OutputMapping(Contract):
    """JSON selectors frozen before execution; every value comes from runtime output."""

    root_path: list[str | int] = Field(default_factory=list)
    dataset_path: list[str | int] = Field(default_factory=lambda: ["dataset"])
    settings_path: list[str | int] | None = Field(default_factory=lambda: ["settings"])
    settings_paths: dict[str, list[str | int]] = Field(default_factory=dict)
    metric_paths: dict[str, list[str | int]] = Field(default_factory=dict)


class ReleasedArtifactContract(Contract):
    """Operator-approved role and selectors, supplied before any run output exists."""

    artifact_path: NonEmpty
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    artifact_kind: Literal["data", "logs"]
    values_key: NonEmpty
    dataset_key: NonEmpty
    metric_key: NonEmpty
    settings_key: NonEmpty
    operation: Literal["mean", "sum", "min", "max", "identity"]


class PaperVariance(Contract):
    value: Annotated[FiniteNumber, Field(ge=0)]
    block_id: NonEmpty
    quote: NonEmpty


SANDBOX_ENV = {"HOME": "/workspace/run_dir/home", "TMPDIR": "/workspace/run_dir/.tmp"}
SANDBOX_PAIR_FLAGS = {"--cpus", "--memory", "--memory-swap", "--pids-limit", "--network",
                      "--cap-drop", "--security-opt"}


class SandboxPolicy(Contract):
    """Operator-only commercial runtime limits; scientific configuration stays separate."""

    version: Literal["commercial-sandbox-v1"] = "commercial-sandbox-v1"
    cpus: Annotated[FiniteNumber, Field(gt=0, strict=True)] = 2.0
    memory_bytes: int = Field(default=4294967296, gt=0, strict=True)
    pids_limit: int = Field(default=256, gt=0, strict=True)
    network: str = Field(default="none", min_length=1, max_length=64, strict=True)
    uid: int = Field(default=1000, gt=0, strict=True)
    gid: int = Field(default=1000, gt=0, strict=True)
    read_only_rootfs: bool = Field(default=True, strict=True)
    cap_drop: Literal["ALL"] = "ALL"
    no_new_privileges: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def commercial_boundary(self):
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.network)
                or self.network.lower() in {"host", "container", "default"}):
            raise ValueError("Sandbox network must be none, bridge or an operator-preconfigured named network")
        if not self.read_only_rootfs or not self.no_new_privileges:
            raise ValueError("Commercial sandbox requires readonly rootfs and no-new-privileges")
        return self


def sandbox_policy(value):
    # Reparse mutable model instances instead of trusting model_copy/model_construct.
    return SandboxPolicy.model_validate(value.model_dump(mode="json") if isinstance(value, SandboxPolicy) else value)


def sandbox_options(value):
    policy = sandbox_policy(value)
    return {"--cpus": str(policy.cpus), "--memory": str(policy.memory_bytes),
            "--memory-swap": str(policy.memory_bytes), "--pids-limit": str(policy.pids_limit),
            "--network": policy.network, "--user": f"{policy.uid}:{policy.gid}",
            "--cap-drop": policy.cap_drop, "--security-opt": "no-new-privileges=true"}


def validate_sandbox_options(options, value):
    """Check the finite actual Docker prefix; duplicate/overridden limits fail closed."""
    expected = sandbox_options(value)
    pairs = {"--name", "--user", "--gpus", "--shm-size", "--ipc", "-w", "--workdir", "-e", "--env",
             "-v", "--volume", "--mount"} | SANDBOX_PAIR_FLAGS
    seen, environment, readonly, index = set(), {}, 0, 0
    ipc_count = 0
    while index < len(options):
        option = options[index]
        if option in {"--rm", "--read-only"}:
            readonly += option == "--read-only"
            index += 1
            continue
        if option not in pairs or index + 1 >= len(options):
            raise ValueError("sandbox Docker option layout unavailable")
        content = options[index + 1]
        if option == "--ipc":
            ipc_count += 1
            if ipc_count != 1 or content != "private":
                raise ValueError("sandbox IPC must be private and unique")
        if option in expected:
            if option in seen or content != expected[option]:
                raise ValueError("sandbox Docker limit changed or duplicated")
            seen.add(option)
        if option in {"-e", "--env"}:
            key, separator, item = content.partition("=")
            if key in SANDBOX_ENV:
                if key in environment or not separator or item != SANDBOX_ENV[key]:
                    raise ValueError("sandbox writable environment changed")
                environment[key] = item
        index += 2
    if seen != set(expected) or readonly != 1 or environment != SANDBOX_ENV:
        raise ValueError("sandbox Docker limits or writable environment missing")


class ExecutionConfig(Contract):
    max_attempts: int = Field(default=3, ge=0, le=3, description="Maximum accepted repair rounds")
    approval_mode: Literal["auto", "interactive"] = "auto"
    training_budget: int = Field(default=0, ge=0, description="Maximum training runs, including retries")
    timeout_seconds: int = Field(default=3600, gt=0)
    python_version: str = "3.11"
    docker_build_timeout_seconds: int = Field(default=3600, gt=0)
    docker_options: dict[str, Any] = Field(default_factory=dict)
    sandbox: SandboxPolicy = Field(default_factory=SandboxPolicy)
    refine_with_llm: bool = True
    output_mappings: dict[str, OutputMapping] = Field(default_factory=dict)
    author_artifacts: dict[str, list[ReleasedArtifactContract]] = Field(default_factory=dict)
    paper_variances: dict[str, dict[str, PaperVariance]] = Field(default_factory=dict)
    tolerance_overrides: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )


class TorchInferenceLimits(Contract):
    """Operator policy, supplied independently of author/model proposals."""

    protocol: Literal["torch-cpu-inference-v1"] = "torch-cpu-inference-v1"
    torch_version: Literal["2.14.0"] = "2.14.0"
    # No guessed or downloaded environment. A real deployment must pin these.
    image_id: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    package_manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    max_tensor_elements: int = Field(default=1048576, ge=1, le=1048576)
    max_events: int = Field(default=8192, ge=1, le=65536)
    max_samples: int = Field(default=100000, ge=1, le=1000000)
    max_jsonl_line_bytes: int = Field(default=1048576, ge=1024, le=1048576)
    max_partition_bytes: int = Field(default=33554432, ge=1024, le=33554432)
    max_report_bytes: int = Field(default=33554432, ge=1024, le=33554432)
