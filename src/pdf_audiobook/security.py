"""Small, explicit security and instance-state primitives for the local app."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

INSTANCE_SCHEMA = 1
DEFAULT_PORT_START = 8765
DEFAULT_PORT_END = 8774
TOKEN_BYTES = 32


def data_root() -> Path:
    override = os.environ.get("PDF_AUDIOBOOK_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PDFToAudiobook"
    return Path.home() / ".pdf-audiobook"


def instance_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "instance.json"


def lock_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "instance.lock"


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def new_session_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def choose_port(start: int = DEFAULT_PORT_START, end: int = DEFAULT_PORT_END) -> int:
    if not (1 <= start <= end <= 65535):
        raise ValueError("invalid port range")
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"no available loopback port in {start}-{end}")


def build_instance(*, pid: int, port: int, launch_id: str, token: str) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("invalid pid")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("invalid port")
    if not launch_id or not token:
        raise ValueError("launch identity and token are required")
    return {
        "schema_version": INSTANCE_SCHEMA,
        "pid": pid,
        "port": port,
        "launch_id": launch_id,
        "session_token": token,
        "session_token_sha256": token_hash(token),
        "started_at": time.time(),
    }


def validate_instance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("instance state must be an object")
    required = {"schema_version", "pid", "port", "launch_id", "session_token", "session_token_sha256", "started_at"}
    if set(value) != required:
        raise ValueError("instance state schema mismatch")
    if value["schema_version"] != INSTANCE_SCHEMA:
        raise ValueError("unsupported instance schema")
    if not isinstance(value["pid"], int) or value["pid"] <= 0:
        raise ValueError("invalid instance pid")
    if not isinstance(value["port"], int) or not 1024 <= value["port"] <= 65535:
        raise ValueError("invalid instance port")
    if not isinstance(value["launch_id"], str) or not value["launch_id"]:
        raise ValueError("invalid launch identity")
    if not isinstance(value["session_token"], str) or len(value["session_token"]) < 32:
        raise ValueError("invalid session token")
    if value["session_token_sha256"] != token_hash(value["session_token"]):
        raise ValueError("instance token checksum mismatch")
    if not isinstance(value["started_at"], (int, float)) or value["started_at"] <= 0:
        raise ValueError("invalid start time")
    return value


def read_instance(path: Path | None = None) -> dict[str, Any] | None:
    target = path or instance_path()
    try:
        if target.is_symlink() or not target.is_file():
            return None
        return validate_instance(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def atomic_write_instance(value: dict[str, Any], path: Path | None = None) -> Path:
    target = path or instance_path()
    validate_instance(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(temp, target)
        try:
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)
    return target


def remove_instance_if_matches(path: Path, *, launch_id: str, pid: int, token: str) -> bool:
    """Unlink only our regular instance file; never follow a symlink."""
    try:
        if path.is_symlink() or not path.is_file():
            return False
        current = read_instance(path)
        if not current or current["launch_id"] != launch_id or current["pid"] != pid or current["session_token"] != token:
            return False
        path.unlink()
        return True
    except OSError:
        return False


def _windows_pid_is_alive(pid: int) -> bool:
    """Check a Windows process without requesting terminate rights."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    try:
        try:
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        except (OSError, OverflowError, ValueError):
            return False
        if not handle:
            return ctypes.get_last_error() == error_access_denied
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return ctypes.get_last_error() == error_access_denied
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, OverflowError, ValueError):
        return False


def pid_is_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OSError, OverflowError, ValueError):
        return False
    return True
