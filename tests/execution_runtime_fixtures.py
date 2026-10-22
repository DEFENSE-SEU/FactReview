"""Mock host captures obey the same operator sandbox contract as native runs."""

from fact_generation.execution.config import SANDBOX_ENV, sandbox_options


def sandbox_argv(argv, request):
    options = ["--read-only"]
    for flag, value in sandbox_options(request.config.sandbox).items():
        options.extend([flag, value])
    for key, value in SANDBOX_ENV.items():
        options.extend(["-e", f"{key}={value}"])
    return [*argv[:2], *options, *argv[2:]]


def sandbox_audit(argv, request):
    policy = request.config.sandbox
    return {"policy": policy.model_dump(mode="json"), "user": f"{policy.uid}:{policy.gid}",
            "network": policy.network, "author_transport_invoked": True, "argv": argv,
            "runtime_enforcement": "unverified"}
