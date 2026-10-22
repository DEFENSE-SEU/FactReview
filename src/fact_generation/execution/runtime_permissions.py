"""Provision host-created mounts for the approved non-root Linux identity."""

import os
import stat
import sys
from pathlib import Path

from .repair_evidence import plain_path


def prepare_runtime_permissions(
    *, workspace, runtime_dir, sandbox, trusted_dir=None, audit_output=None, metric_capture=None
):
    """Adjust private snapshots and explicit output slots before author launch.

    Docker Desktop manages host mounts on Windows/macOS. Native Linux retains
    host ownership, including when the host process runs as root with umask 077.
    Never walk prior author output or grant group/other write permissions.
    """
    if sys.platform != "linux":
        return {"status": "desktop_managed"}
    uid, gid = sandbox.uid, sandbox.gid
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
        raise ValueError("Runtime mount ownership requires a positive uid and gid")
    if os.geteuid() not in (0, uid):
        raise ValueError("Set sandbox.uid to the host user ID, or provision execution as root")

    # Collect and validate every target before changing any permission. Only
    # the private input snapshot and generated observer package are traversed.
    paths = {}
    for root in (workspace, trusted_dir):
        if root is None:
            continue
        root = plain_path(Path(root))
        paths[root] = False
        for directory, directories, files in os.walk(root, followlinks=False):
            for name in (*directories, *files):
                paths[plain_path(Path(directory) / name)] = False

    scratch = plain_path(Path(runtime_dir))
    for path in (scratch, scratch / "home", scratch / ".tmp"):
        paths[plain_path(path)] = True
    for path in (Path(audit_output).parent if audit_output else None, metric_capture):
        if path is not None:
            target = plain_path(Path(path))
            if not target.is_relative_to(scratch) or target == scratch:
                raise ValueError("Runtime output permissions must stay inside scratch")
            paths[target] = True

    targets = []
    for path, writable in paths.items():
        info = path.lstat()
        is_directory = stat.S_ISDIR(info.st_mode)
        if not is_directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError("Runtime mounts require unlinked regular files and directories")
        mode = stat.S_IMODE(info.st_mode) & ~(stat.S_ISUID | stat.S_ISGID)
        mode |= stat.S_IRUSR | (stat.S_IXUSR if is_directory else 0)
        if writable:
            mode |= stat.S_IWUSR
        targets.append((path, info, mode))
    for path, info, mode in targets:
        if (info.st_uid, info.st_gid) != (uid, gid):
            os.chown(path, uid, gid, follow_symlinks=False)
        if stat.S_IMODE(info.st_mode) != mode:
            path.chmod(mode)
    return {"status": "ready", "uid": uid, "gid": gid, "paths": len(targets)}
