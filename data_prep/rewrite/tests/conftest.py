"""Shared test setup: repo root on sys.path, no real network, never a real API key."""

import ipaddress
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_prep.rewrite.tests import synth  # noqa: E402

FAKE_API_KEY = "nvapi-test-not-a-real-key"


def _is_local(host) -> bool:
    if host in (None, "", "localhost") or isinstance(host, bytes) and host == b"localhost":
        return True
    try:
        return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else str(host)).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network_no_key(monkeypatch):
    """Block non-loopback DNS/connects and shadow NVIDIA_API_KEY so a test can never reach or bill the real endpoint."""
    real_getaddrinfo = socket.getaddrinfo
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(address):
        if isinstance(address, tuple) and not _is_local(address[0]):
            raise RuntimeError(f"tests must not use the network (tried {address[0]!r}); use httpx.MockTransport")

    def getaddrinfo(host, *args, **kwargs):
        guard((host,))
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_API_KEY)


@pytest.fixture
def manifest_path(tmp_path) -> Path:
    """Synthetic manifest (default synth.make_manifest settings) with images under tmp_path/data."""
    return synth.make_manifest(tmp_path / "data")
