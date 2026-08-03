"""Per-source acquisition planning.

Each first-wave source is reached a different way, so this module turns a source specification and a
few parameters into the list of acquisition items the driver stores. Anonymous public S3 and public
GCS buckets are listed; the OOI archive is read one named directory at a time; MarineCadastre AIS is
built from a date range; and ONC is listed through its credentialed archive-file API with the token
kept out of the recorded provenance URL. Explicit object URLs are always accepted, which is how a
source without a lister (for example a DCLDE set) is acquired.
"""

from __future__ import annotations

from collections.abc import Sequence

from .acquire import AcquisitionItem, plan_anon_s3_items
from .ais import marinecadastre_daily_urls
from .fetch import anon_s3_list
from .gcs import gcs_list, gcs_object_url
from .onc import onc_download_url, onc_list_archive_files, onc_origin_url, onc_token_from_env
from .ooi import ooi_list_index
from .sources import AccessClass, SourceSpec


class PlanningError(RuntimeError):
    """Raised when a source cannot be planned from the given parameters."""


def _https_item(
    url: str, origin: str | None = None, media_name: str | None = None
) -> AcquisitionItem:
    return AcquisitionItem(origin_url=origin or url, url=url, media_name=media_name)


def plan_items(
    spec: SourceSpec,
    *,
    prefix: str = "",
    limit: int = 1,
    urls: Sequence[str] | None = None,
    index_url: str | None = None,
    ais_start: str | None = None,
    ais_end: str | None = None,
    onc_location: str | None = None,
    onc_device_category: str | None = None,
    onc_extension: str | None = None,
) -> list[AcquisitionItem]:
    """Return the acquisition items for a source, using the appropriate access method."""
    if urls:
        return [_https_item(url) for url in urls]

    if spec.access_class == AccessClass.ANON_S3:
        if spec.s3_bucket is None:
            raise PlanningError(f"{spec.source_id} has no S3 bucket")
        keys = anon_s3_list(spec.s3_bucket, prefix, spec.s3_region or "us-east-1", max_keys=limit)
        return plan_anon_s3_items(spec, keys)

    if spec.access_class == AccessClass.GCS_HTTPS:
        if spec.s3_bucket is None:
            raise PlanningError(f"{spec.source_id} has no GCS bucket")
        names = gcs_list(spec.s3_bucket, prefix, max_keys=limit)
        return [_https_item(gcs_object_url(spec.s3_bucket, name)) for name in names]

    if spec.access_class == AccessClass.HTTPS:
        if index_url is not None:
            return [_https_item(url) for url in ooi_list_index(index_url)[:limit]]
        if ais_start is not None and ais_end is not None:
            return [_https_item(url) for url in marinecadastre_daily_urls(ais_start, ais_end)]
        raise PlanningError(
            f"{spec.source_id} needs explicit --url(s), an --index-url, or an AIS date range"
        )

    if spec.access_class == AccessClass.CREDENTIALED:
        if not (onc_location and onc_device_category and ais_start and ais_end):
            raise PlanningError(
                f"{spec.source_id} needs --onc-location, --onc-device-category, and a date range"
            )
        token = onc_token_from_env()
        names = onc_list_archive_files(
            location_code=onc_location,
            device_category_code=onc_device_category,
            date_from=ais_start,
            date_to=ais_end,
            token=token,
            extension=onc_extension,
        )
        return [
            _https_item(onc_download_url(name, token), origin=onc_origin_url(name), media_name=name)
            for name in names
        ]

    raise PlanningError(f"{spec.source_id} access class {spec.access_class} is not supported here")
