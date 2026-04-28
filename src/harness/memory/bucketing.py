from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta


def floor_to_5_minutes(dt: datetime) -> datetime:
    local = dt.replace(second=0, microsecond=0)
    bucket_minute = (local.minute // 5) * 5
    return local.replace(minute=bucket_minute)


def iter_5_min_buckets(start_inclusive: datetime, end_exclusive: datetime) -> Iterable[datetime]:
    current = floor_to_5_minutes(start_inclusive)
    while current < end_exclusive:
        yield current
        current = current + timedelta(minutes=5)


def hour_start(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def last_completed_5m_end(now: datetime) -> datetime:
    local = now.replace(second=0, microsecond=0)
    floored = floor_to_5_minutes(local)
    if floored == local:
        return local - timedelta(minutes=5)
    return floored
