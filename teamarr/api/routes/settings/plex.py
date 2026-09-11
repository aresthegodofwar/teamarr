"""Plex settings, connection test, and DVR/device discovery endpoints."""

from fastapi import APIRouter

from teamarr.database import get_db
from teamarr.plex.client import PlexClient

from .models import (
    MASKED_SECRET,
    PlexConnectionTestRequest,
    PlexConnectionTestResponse,
    PlexDeviceModel,
    PlexDvrModel,
    PlexDvrsResponse,
    PlexSettingsModel,
    PlexSettingsUpdate,
    merge_masked_servers,
    to_model,
)

router = APIRouter()


@router.get("/settings/plex", response_model=PlexSettingsModel)
def get_plex_settings():
    """Get Plex integration settings."""
    from teamarr.database.settings import get_plex_settings

    with get_db() as conn:
        settings = get_plex_settings(conn)

    return to_model(PlexSettingsModel, settings)


@router.put("/settings/plex", response_model=PlexSettingsModel)
def update_plex_settings(update: PlexSettingsUpdate):
    """Update Plex integration settings."""
    from teamarr.database.settings import get_plex_settings, update_plex_settings

    servers = None
    if update.servers is not None:
        with get_db() as conn:
            stored = get_plex_settings(conn).servers
        servers = merge_masked_servers([s.model_dump() for s in update.servers], stored)

    with get_db() as conn:
        update_plex_settings(conn, enabled=update.enabled, servers=servers)

    with get_db() as conn:
        settings = get_plex_settings(conn)

    return to_model(PlexSettingsModel, settings)


def _resolve_url_token(
    request: PlexConnectionTestRequest | None,
) -> tuple[str, str, str | None]:
    """Resolve (url, token, error) from an optional request against saved settings.

    Masked tokens ("********") are resolved against the saved server whose
    URL matches the request (falling back to the first configured server),
    same pattern as Emby/Jellyfin.
    """
    from teamarr.database.settings import get_plex_settings

    with get_db() as conn:
        saved = get_plex_settings(conn)

    first = saved.servers[0] if saved.servers else None
    match = None
    if request and request.url:
        match = next((s for s in saved.servers if s.url == request.url), first)
    token = request.token if request else None
    if token == MASKED_SECRET:
        token = (match or first).token if (match or first) else None

    url = (request.url if request and request.url else (first.url if first else None)) or ""
    token = token or ((match or first).token if (match or first) else None) or ""
    return url, token, None


@router.post("/plex/test", response_model=PlexConnectionTestResponse)
def test_plex_connection(request: PlexConnectionTestRequest | None = None):
    """Test connection to a Plex server.

    If no parameters provided, tests with the first saved server.
    """
    url, token, _ = _resolve_url_token(request)
    if not url:
        return PlexConnectionTestResponse(success=False, error="No Plex URL configured")
    if not token:
        return PlexConnectionTestResponse(success=False, error="No Plex token configured")

    client = PlexClient(base_url=url, token=token)
    result = client.test_connection()

    return PlexConnectionTestResponse(
        success=result.get("success", False),
        dvr_count=result.get("dvr_count"),
        error=result.get("error"),
    )


@router.get("/plex/dvrs", response_model=PlexDvrsResponse)
def list_plex_dvrs(url: str | None = None, token: str | None = None):
    """List DVRs and their HDHomeRun devices, for the device picker.

    Falls back to the saved URL/token when not provided. A masked token
    ("********") resolves against the saved server matching `url`.
    """
    request = PlexConnectionTestRequest(url=url, token=token) if (url or token) else None
    resolved_url, resolved_token, _ = _resolve_url_token(request)

    if not resolved_url:
        return PlexDvrsResponse(success=False, error="No Plex URL configured")
    if not resolved_token:
        return PlexDvrsResponse(success=False, error="No Plex token configured")

    client = PlexClient(base_url=resolved_url, token=resolved_token)
    result = client.list_dvrs()

    if not result.get("success"):
        return PlexDvrsResponse(success=False, error=result.get("error"))

    dvrs = [
        PlexDvrModel(
            key=dvr.key,
            lineup_title=dvr.lineup_title,
            devices=[
                PlexDeviceModel(
                    key=device.key,
                    device_id=device.device_id,
                    uri=device.uri,
                    profile_hint=device.profile_hint,
                    channel_count=len(
                        [m for m in device.channel_mapping if m.enabled]
                    ),
                )
                for device in dvr.devices
            ],
        )
        for dvr in result["dvrs"]
    ]
    return PlexDvrsResponse(success=True, dvrs=dvrs)
