"""Plex Media Server client for Live TV guide + channel-map refresh.

Plex's Live TV & DVR feature (pointed at Dispatcharr's HDHomeRun emulation)
does not notice new/removed channels on its own — two calls are needed after
each Teamarr generation:

* ``POST /livetv/dvrs/<dvr_id>/reloadGuide`` — refreshes programme/EPG data
  for channels Plex already knows about. Does NOT add new channels.
* ``PUT /media/grabbers/devices/<device_key>/channelmap`` — the enable/EPG-bind
  call. This is a **full-state-replace**, not additive: Plex Web always sends
  the entire desired enabled-channel set, never a delta. An integration that
  PUTs only its own channels would silently disable every other channel on
  that device (other tools, manually-added channels, etc.) — see
  ``compute_channelmap_update``, which preserves everything outside
  Teamarr's own channel-number range.

Both endpoints require ``X-Plex-Token`` (sent as a header here) — no other
auth scheme. ``GET /livetv/dvrs`` is the read side: it returns every DVR and
its attached HDHomeRun devices, each carrying its current ``ChannelMapping``
(channelKey/enabled/lineupIdentifier/deviceIdentifier) — the fetch half of
the fetch-merge-write cycle.

Verified against a live server (2026-09): ``deviceIdentifier`` is the
stable Dispatcharr/HDHomeRun physical channel number and never changes.
``channelKey`` is whatever EPG entry is *currently matched* in Plex's
Channel Matching UI — it starts out equal to ``deviceIdentifier`` but
diverges the moment that match is changed (manually, or by Plex's own
auto-matching). Everything here keys on ``deviceIdentifier``; ``channelKey``
is kept only for display/debugging and must never drive an ownership or
identity decision.
"""

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PlexChannelMapping:
    """One tuner-channel entry on a Plex DVR device.

    ``device_identifier`` is the stable Dispatcharr/HDHomeRun physical
    channel number — the identity everything is keyed on. ``channel_key``
    is Plex's currently-matched EPG guide channel (mutable, display-only —
    never use it for ownership/identity, see module docstring).
    ``lineup_identifier`` is the EPG binding to preserve for a foreign
    channel that isn't Teamarr's own.
    """

    device_identifier: str
    enabled: bool
    lineup_identifier: str | None = None
    channel_key: str | None = None


@dataclass
class PlexDevice:
    """One HDHomeRun (emulated) device attached to a Plex DVR."""

    key: str
    device_id: str | None = None
    uri: str | None = None
    channel_mapping: list[PlexChannelMapping] = field(default_factory=list)

    @property
    def profile_hint(self) -> str | None:
        """Last path segment of ``uri`` (e.g. the Dispatcharr profile name).

        None for root-scoped devices (``.../hdhr`` with no suffix, or no
        URI at all) — those pull every channel from every profile and need
        manual confirmation rather than an automatic match.
        """
        if not self.uri:
            return None
        path = urlsplit(self.uri).path.rstrip("/")
        segment = path.rsplit("/", 1)[-1] if path else ""
        return segment if segment and segment.lower() != "hdhr" else None


@dataclass
class PlexDvr:
    """One DVR configured in Plex Live TV."""

    key: str
    lineup_title: str | None = None
    lineup: str | None = None
    devices: list[PlexDevice] = field(default_factory=list)


def _as_list(value: Any) -> list:
    """Normalize a Plex JSON container that may collapse to a bare object.

    Plex-style APIs are known to return a single-child container as one
    object instead of a 1-element array (e.g. one DVR, one device, one
    channel mapping) — unverified here against a live server, but cheap
    to guard against regardless.
    """
    if isinstance(value, dict):
        return [value]
    return list(value) if value else []


def _channel_sort_key(channel_key: str) -> tuple[int, object]:
    try:
        return (0, int(float(channel_key)))
    except (TypeError, ValueError):
        return (1, channel_key)


def compute_channelmap_update(
    current: list[PlexChannelMapping],
    teamarr_channel_keys: set[str],
    teamarr_range: tuple[int, int | None] | None,
) -> tuple[list[str], dict[str, str]]:
    """Fetch-merge-write core: compute the full PUT payload for one device.

    ``current`` is the device's existing ``ChannelMapping`` (from
    ``PlexClient.list_dvrs``). Every currently-enabled channel OUTSIDE
    ``teamarr_range`` is preserved untouched — this is what protects
    other tools' or manually-added channels sharing the same device,
    especially on a root-scoped device pulling multiple profiles.
    ``teamarr_range`` is ``(start, end)`` with ``end=None`` meaning
    unbounded; pass ``None`` only when Teamarr manages no range at all
    (nothing is preserved-vs-owned in that case — every enabled channel
    is treated as foreign).

    Everything is keyed on ``device_identifier`` (the stable Dispatcharr
    physical channel number), never ``channel_key`` (Plex's mutable,
    currently-matched EPG guide channel — see module docstring). A
    preserved foreign channel's value is its own current
    ``lineup_identifier``, so an existing (possibly non-default) EPG
    binding is carried forward untouched rather than overwritten.

    Returns ``(enabled_channel_keys, channel_mapping)`` ready for
    ``PlexClient.update_channelmap``. Teamarr's own channels map to
    themselves (``channel_mapping[key] == key``) — correct as long as
    Dispatcharr/XMLTV numbering is aligned (Teamarr's documented default
    setup); see the Plex settings docs for the numbering-alignment
    prerequisite.
    """

    def _owned_by_teamarr(device_identifier: str) -> bool:
        if teamarr_range is None:
            return False
        start, end = teamarr_range
        try:
            number = int(float(device_identifier))
        except (TypeError, ValueError):
            return False
        if end is None:
            return number >= start
        return start <= number <= end

    preserved = [
        m for m in current if m.enabled and not _owned_by_teamarr(m.device_identifier)
    ]

    mapping: dict[str, str] = {
        m.device_identifier: m.lineup_identifier or m.device_identifier for m in preserved
    }
    for key in teamarr_channel_keys:
        mapping[key] = key

    enabled = sorted(mapping, key=_channel_sort_key)
    return enabled, mapping


class PlexClient:
    """Client for the Plex Media Server Live TV / DVR API."""

    SERVER_LABEL: str = "PLEX"

    def __init__(self, base_url: str, token: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-Plex-Token": self.token, "Accept": "application/json"}

    def list_dvrs(self) -> dict:
        """GET /livetv/dvrs — every configured DVR with its devices/channel maps."""
        try:
            resp = httpx.get(
                f"{self.base_url}/livetv/dvrs",
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            message = "Invalid Plex token" if status == 401 else f"HTTP {status}"
            return {"success": False, "error": message}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e)}

        container = (data or {}).get("MediaContainer") or {}
        dvrs: list[PlexDvr] = []
        for dvr in _as_list(container.get("Dvr")):
            devices: list[PlexDevice] = []
            for dev in _as_list(dvr.get("Device")):
                mapping = [
                    PlexChannelMapping(
                        device_identifier=str(m.get("deviceIdentifier")),
                        enabled=str(m.get("enabled")) == "1",
                        lineup_identifier=m.get("lineupIdentifier"),
                        channel_key=(
                            str(m.get("channelKey")) if m.get("channelKey") is not None else None
                        ),
                    )
                    for m in _as_list(dev.get("ChannelMapping"))
                ]
                devices.append(
                    PlexDevice(
                        key=str(dev.get("key")),
                        device_id=dev.get("deviceId"),
                        uri=dev.get("uri"),
                        channel_mapping=mapping,
                    )
                )
            dvrs.append(
                PlexDvr(
                    key=str(dvr.get("key")),
                    lineup_title=dvr.get("lineupTitle"),
                    lineup=dvr.get("lineup"),
                    devices=devices,
                )
            )
        return {"success": True, "dvrs": dvrs}

    def test_connection(self) -> dict:
        """Verify the URL/token combination works."""
        result = self.list_dvrs()
        if not result["success"]:
            return result
        return {"success": True, "dvr_count": len(result["dvrs"])}

    def reload_guide(self, dvr_id: str) -> dict:
        """POST /livetv/dvrs/<dvr_id>/reloadGuide — refresh programme data.

        Does not add/remove channels — see ``update_channelmap`` for that.
        """
        try:
            resp = httpx.post(
                f"{self.base_url}/livetv/dvrs/{dvr_id}/reloadGuide",
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e)}
        return {"success": True}

    def update_channelmap(
        self,
        device_key: str,
        enabled_channel_keys: list[str],
        channel_mapping: dict[str, str],
    ) -> dict:
        """PUT /media/grabbers/devices/<device_key>/channelmap — enable + EPG-bind.

        Full-state-replace: ``enabled_channel_keys`` must be the COMPLETE
        desired enabled set (see ``compute_channelmap_update``), not a delta.
        Both `channelMappingByKey` and `channelMapping` are keyed by the
        physical channel (device identifier) and carry the same resolved
        EPG-binding value — previously `channelMappingByKey` always sent
        the key back as its own value regardless of `channel_mapping`,
        which disagreed with `channelMapping` for a preserved foreign
        channel whose EPG binding differs from its channel number.
        """
        params: list[tuple[str, str]] = [
            ("channelsEnabled", ",".join(enabled_channel_keys)),
        ]
        for key, value in channel_mapping.items():
            params.append((f"channelMappingByKey[{key}]", value))
            params.append((f"channelMapping[{key}]", value))

        try:
            resp = httpx.put(
                f"{self.base_url}/media/grabbers/devices/{device_key}/channelmap",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "channel_count": len(enabled_channel_keys)}
