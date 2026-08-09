"""Stale-safe, single-instance launcher for the localhost application."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Callable

import uvicorn

from .app import create_app
from .security import (
    DEFAULT_PORT_END,
    DEFAULT_PORT_START,
    atomic_write_instance,
    build_instance,
    choose_port,
    data_root,
    instance_path,
    lock_path,
    pid_is_alive,
    read_instance,
    remove_instance_if_matches,
)


class InstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.marker = ""

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = open(self.path, "x", encoding="ascii")
            self.marker = f"{os.getpid()}:{uuid.uuid4().hex}"
            self.handle.write(self.marker)
            self.handle.flush()
            return True
        except FileExistsError:
            # A crashed launcher can leave only the lock marker behind. Remove
            # that exact regular file when its recorded owner is gone; never
            # follow a symlink while recovering stale state.
            try:
                if self.path.is_symlink():
                    return False
                marker = self.path.read_text(encoding="ascii").strip()
                owner_text, _, _ = marker.partition(":")
                owner = int(owner_text)
                if not pid_is_alive(owner):
                    self.path.unlink()
                    return self.acquire()
            except (OSError, ValueError):
                pass
            return False

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.close()
            self.handle = None
            if self.path.is_symlink() or not self.path.is_file():
                return
            if self.path.read_text(encoding="ascii").strip() == self.marker:
                self.path.unlink()
        except OSError:
            pass


def _health(instance: dict, timeout: float = 0.35) -> bool:
    request = urllib.request.Request(
        f"http://127.0.0.1:{instance['port']}/health",
        headers={"Host": f"127.0.0.1:{instance['port']}", "X-Instance-Token": instance["session_token"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value.get("ready") is True and value.get("launch_id") == instance["launch_id"]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def existing_instance(root: Path) -> dict | None:
    instance = read_instance(instance_path(root))
    if instance and pid_is_alive(instance["pid"]) and _health(instance):
        return instance
    return None


def _open_url(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:
        return False


def run_launcher(*, root: Path | None = None, open_browser: Callable[[str], bool] = _open_url, wait_timeout: float = 10.0, run_server: bool = True) -> int:
    root = root or data_root()
    root.mkdir(parents=True, exist_ok=True)
    lock = InstanceLock(lock_path(root))
    if not lock.acquire():
        for _ in range(40):
            current = existing_instance(root)
            if current:
                url = f"http://127.0.0.1:{current['port']}/#session={current['session_token']}"
                if not open_browser(url):
                    print(f"PDF Audiobook is already running at http://127.0.0.1:{current['port']}/")
                return 0
            time.sleep(0.05)
        return 1

    instance_file = instance_path(root)
    launch_id = uuid.uuid4().hex
    token = __import__("secrets").token_urlsafe(32)
    instance_written = False
    server = None
    thread = None
    try:
        stale = read_instance(instance_file)
        if stale:
            remove_instance_if_matches(instance_file, launch_id=stale["launch_id"], pid=stale["pid"], token=stale["session_token"])
        port = choose_port(DEFAULT_PORT_START, DEFAULT_PORT_END)
        instance = build_instance(pid=os.getpid(), port=port, launch_id=launch_id, token=token)
        atomic_write_instance(instance, instance_file)
        instance_written = True
        app = create_app(port=port, launch_id=launch_id, session_token=token, instance_file=instance_file)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="warning")
        server = uvicorn.Server(config)
        app.state.phase1.uvicorn_server = server
        thread = threading.Thread(target=server.run, name="pdf-audiobook-server", daemon=True)
        thread.start()
        deadline = time.monotonic() + wait_timeout
        ready = False
        while time.monotonic() < deadline:
            if _health(instance):
                ready = True
                break
            time.sleep(0.05)
        if not ready:
            return 1
        if not open_browser(f"http://127.0.0.1:{port}/#session={token}"):
            print(f"PDF Audiobook is ready at http://127.0.0.1:{port}/")
        if not run_server:
            server.should_exit = True
            thread.join(timeout=3)
        else:
            while not app.state.phase1.shutdown_event.wait(0.1):
                if not thread.is_alive():
                    break
            server.should_exit = True
            thread.join(timeout=3)
        return 0
    finally:
        if server is not None and thread is not None and thread.is_alive():
            server.should_exit = True
            thread.join(timeout=3)
        if instance_written:
            remove_instance_if_matches(instance_file, launch_id=launch_id, pid=os.getpid(), token=token)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the private PDF-to-audiobook localhost app")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opener = (lambda _url: False) if args.no_browser else _open_url
    return run_launcher(open_browser=opener)
