from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .fs import ensure_dir, write_text
from .verbose import is_verbose


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    # None preserves the unknown origin of historical/uninstrumented results.
    termination: Literal[
        "completed", "launch_failed", "communication_failed", "timed_out", "unfinished"
    ] | None = None
    exception_type: str | None = None


def _tail(text: str, n: int = 2000) -> str:
    s = text or ""
    if len(s) <= n:
        return s
    return s[-n:]


def _create_windows_kill_on_close_job(proc: subprocess.Popen[str]) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JobObjectBasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JobObjectExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JobObjectBasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            kernel32.CloseHandle(job)
            return None
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            kernel32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _close_windows_job(job_handle: int | None) -> None:
    if os.name != "nt" or not job_handle:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(job_handle))
    except Exception:
        pass


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        if os.name == "nt":
            ps_list_script = f"""
$ErrorActionPreference = 'SilentlyContinue'
$root = {int(proc.pid)}
$front = @($root)
$all = @()
while ($front.Count -gt 0) {{
    $children = Get-CimInstance Win32_Process | Where-Object {{ $front -contains $_.ParentProcessId }}
    $ids = @($children | ForEach-Object {{ [int]$_.ProcessId }})
    $all += $ids
    $front = $ids
}}
(($all + $root) | Select-Object -Unique) -join "`n"
"""
            pids = [int(proc.pid)]
            try:
                listed = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_list_script],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                discovered = [int(x.strip()) for x in (listed.stdout or "").splitlines() if x.strip().isdigit()]
                # Kill children before parents, while we still have the original
                # descendant PIDs. Once the parent dies, Windows can re-parent
                # children and parent-id traversal loses them.
                pids = list(dict.fromkeys([*discovered, int(proc.pid)]))
            except Exception:
                pids = [int(proc.pid)]
            for pid in pids:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            ps_stop = " ".join(str(pid) for pid in pids)
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {ps_stop.replace(' ', ',')} -Force -ErrorAction SilentlyContinue",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_command(
    cmd: list[str],
    cwd: str,
    timeout_sec: int | None = 3600,
    env: dict[str, str] | None = None,
) -> CommandResult:
    start = time.time()
    proc = None
    termination, exception_type = "completed", None
    effective_timeout = None if timeout_sec is None or int(timeout_sec) <= 0 else int(timeout_sec)
    if is_verbose():
        try:
            print(
                f"[execution][exec] cwd={cwd} timeout_sec={effective_timeout} cmd={' '.join(cmd)}",
                flush=True,
            )
        except Exception:
            pass
    try:
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        )
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env or os.environ.copy(),
            shell=False,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        windows_job = _create_windows_kill_on_close_job(proc)
        try:
            out, err = proc.communicate(timeout=effective_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            termination, exception_type = "timed_out", type(e).__name__
            _close_windows_job(windows_job)
            windows_job = None
            _kill_process_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                out = e.stdout or ""
                err = e.stderr or ""
            rc = 124
            err = f"TimeoutExpired: command exceeded timeout_sec={effective_timeout}" + (
                f"\n{err}" if str(err or "").strip() else ""
            )
        finally:
            _close_windows_job(windows_job)
        if rc is None:
            rc = 124
            termination = "unfinished"
            _kill_process_tree(proc)
        out = out or ""
        err = err or ""
    except Exception as e:
        # Never crash the workflow due to missing executables (WinError 2), permission issues, etc.
        rc = 127
        out = ""
        err = f"{type(e).__name__}: {e}"
        termination = "launch_failed" if proc is None else "communication_failed"
        exception_type = type(e).__name__
        if proc is not None:
            _kill_process_tree(proc)
    dur = time.time() - start
    if is_verbose():
        try:
            print(f"[execution][exec_done] rc={rc} sec={dur:.3f}", flush=True)
            if rc != 0:
                st = _tail(err or "", 2000).strip()
                if st:
                    print(f"[execution][stderr_tail]\n{st}\n", flush=True)
        except Exception:
            pass
    return CommandResult(
        cmd=cmd,
        cwd=cwd,
        returncode=rc,
        stdout=out,
        stderr=err,
        duration_sec=dur,
        termination=termination,
        exception_type=exception_type,
    )


def persist_command_result(
    result: CommandResult,
    logs_dir: str | Path,
    prefix: str,
) -> None:
    logs = ensure_dir(logs_dir)
    cmd_path = logs / f"{prefix}_command.txt"
    out_path = logs / f"{prefix}_stdout.log"
    err_path = logs / f"{prefix}_stderr.log"
    write_text(
        cmd_path,
        f"cwd: {result.cwd}\ncmd: {' '.join(result.cmd)}\nrc: {result.returncode}\nsec: {result.duration_sec:.3f}\n"
        f"termination: {result.termination or 'unknown'}\nexception_type: {result.exception_type or ''}\n",
    )
    write_text(out_path, result.stdout)
    write_text(err_path, result.stderr)
    if is_verbose():
        try:
            print(
                f"[execution][logs] command={cmd_path} stdout={out_path} stderr={err_path}",
                flush=True,
            )
        except Exception:
            pass
