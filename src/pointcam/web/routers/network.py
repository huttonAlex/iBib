"""Network management endpoints — interface status, WiFi scan/connect via nmcli."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


async def _run(cmd: list[str], timeout: float = 15.0) -> str:
    """Run a subprocess and return stdout. Raises HTTPException on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, "Command timed out")
    except FileNotFoundError:
        raise HTTPException(500, f"Command not found: {cmd[0]}")

    if proc.returncode != 0:
        msg = stderr.decode(errors="replace").strip() or f"exit code {proc.returncode}"
        raise HTTPException(500, msg)

    return stdout.decode(errors="replace").strip()


# ---------------------------------------------------------------------------
# GET /status — list network interfaces
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_status() -> list[dict[str, Any]]:
    """Return all network interfaces with type, state, connection, and IPs."""
    raw = await _run([
        "nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status",
    ])

    # Batch-fetch all IPv4 addresses in one subprocess call (avoids N+1)
    ip_by_device: dict[str, list[str]] = {}
    try:
        ip_raw = await _run(["ip", "-o", "-4", "addr"])
        for ip_line in ip_raw.splitlines():
            tokens = ip_line.split()
            # format: "2: eth0    inet 192.168.1.10/24 ..."
            if len(tokens) >= 4:
                dev = tokens[1]
                for i, tok in enumerate(tokens):
                    if tok == "inet" and i + 1 < len(tokens):
                        ip_by_device.setdefault(dev, []).append(tokens[i + 1].split("/")[0])
    except HTTPException:
        pass  # ip command not available

    interfaces: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split(":")
        if len(parts) < 4:
            continue
        device, dev_type, state, connection = parts[0], parts[1], parts[2], parts[3]
        if dev_type == "loopback":
            continue

        interfaces.append({
            "device": device,
            "type": dev_type,
            "state": state,
            "connection": connection if connection != "--" else None,
            "ips": ip_by_device.get(device, []),
        })

    return interfaces


# ---------------------------------------------------------------------------
# GET /wifi/scan — rescan and list WiFi networks
# ---------------------------------------------------------------------------


@router.get("/wifi/scan")
async def wifi_scan() -> list[dict[str, Any]]:
    """Trigger a WiFi rescan and return visible networks."""
    # Trigger rescan (ignore errors — may fail if no WiFi device)
    try:
        await _run(["nmcli", "device", "wifi", "rescan"], timeout=10.0)
    except HTTPException:
        pass  # rescan can fail if already scanning or no wifi adapter

    try:
        raw = await _run([
            "nmcli", "-t", "-f", "IN-USE,SSID,SIGNAL,SECURITY", "device", "wifi", "list",
        ])
    except HTTPException:
        return []  # no WiFi adapter or nmcli not available

    seen: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        parts = line.split(":")
        if len(parts) < 4:
            continue
        in_use = parts[0].strip() == "*"
        ssid = parts[1].strip()
        if not ssid:
            continue  # hidden networks
        try:
            signal = int(parts[2].strip())
        except ValueError:
            signal = 0
        security = parts[3].strip()

        # Keep strongest signal per SSID
        if ssid not in seen or signal > seen[ssid]["signal"]:
            seen[ssid] = {
                "ssid": ssid,
                "signal": signal,
                "security": security,
                "in_use": in_use,
            }

    # Sort: in-use first, then by signal strength descending
    return sorted(seen.values(), key=lambda n: (-n["in_use"], -n["signal"]))


# ---------------------------------------------------------------------------
# POST /wifi/connect — connect to a WiFi network
# ---------------------------------------------------------------------------


class WifiConnectRequest(BaseModel):
    ssid: str
    password: str = ""


@router.post("/wifi/connect")
async def wifi_connect(req: WifiConnectRequest) -> dict[str, str]:
    """Connect to a WiFi network by SSID."""
    cmd = ["nmcli", "device", "wifi", "connect", req.ssid]
    if req.password:
        cmd += ["password", req.password]

    await _run(cmd, timeout=30.0)
    return {"status": "connected", "ssid": req.ssid}


# ---------------------------------------------------------------------------
# POST /wifi/disconnect — disconnect a network device
# ---------------------------------------------------------------------------


class WifiDisconnectRequest(BaseModel):
    device: str


@router.post("/wifi/disconnect")
async def wifi_disconnect(req: WifiDisconnectRequest) -> dict[str, str]:
    """Disconnect a network device."""
    await _run(["nmcli", "device", "disconnect", req.device])
    return {"status": "disconnected", "device": req.device}
