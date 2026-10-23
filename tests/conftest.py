"""Keep tests offline while allowing Windows asyncio's local connections."""

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is disabled in the offline test suite.")

    def guard(original):
        def connect(sock, address):
            # Windows implements asyncio's socketpair using loopback TCP.
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                try:
                    if ipaddress.ip_address(address[0]).is_loopback:
                        return original(sock, address)
                except ValueError:
                    pass
            return blocked()
        return connect

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(socket.socket.connect_ex))
    monkeypatch.setattr(socket, "create_connection", blocked)
