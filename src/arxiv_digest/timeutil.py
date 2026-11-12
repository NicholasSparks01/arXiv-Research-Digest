"""Collection-window management (America/New_York research weeks)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Window:
    start: datetime  # inclusive, tz-aware
    end: datetime  # exclusive, tz-aware
    research_week_start: date  # Monday of the reported research week
    research_week_end: date  # Sunday of the reported research week
    reason: str

    def arxiv_range(self) -> str:
        """submittedDate filter in arXiv's GMT YYYYMMDDHHMM format (end inclusive to the minute)."""
        s = self.start.astimezone(timezone.utc)
        e = (self.end - timedelta(minutes=1)).astimezone(timezone.utc)
        return f"submittedDate:[{s:%Y%m%d%H%M} TO {e:%Y%m%d%H%M}]"

    def label(self) -> str:
        a, b = self.research_week_start, self.research_week_end
        if a.month == b.month:
            return f"{a:%B} {a.day}–{b.day}, {b.year}"
        return f"{a:%B} {a.day} – {b:%B} {b.day}, {b.year}"


def now_in(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def previous_research_week(now: datetime) -> tuple[date, date]:
    """Monday–Sunday of the last completed week relative to `now` (in now's timezone).

    A Sunday counts as the last day of its own week: the weekly run is scheduled for Sunday morning, and the week
    it reports is the one ending that day. The window therefore ends at the coming Monday 00:00, a few hours in the
    future — arXiv has nothing there yet, and anything submitted later that Sunday is picked up by the next run's
    overlap. On any other weekday the week is the one ending on the previous Sunday, as before.
    """
    today = now.date()
    week_end = today - timedelta(days=(today.weekday() + 1) % 7)  # the most recent Sunday (today, if today is Sunday)
    return week_end - timedelta(days=6), week_end


def compute_window(
    now: datetime,
    tz_name: str,
    last_successful_end: datetime | None,
    initial_lookback_days: int = 30,
    overlap_days: int = 3,
) -> Window:
    tz = ZoneInfo(tz_name)
    now = now.astimezone(tz)
    week_start, week_end = previous_research_week(now)
    end = datetime.combine(week_end + timedelta(days=1), time(0, 0), tz)  # Monday 00:00, exclusive
    week_start_dt = datetime.combine(week_start, time(0, 0), tz)

    if last_successful_end is None:
        start = end - timedelta(days=initial_lookback_days)
        reason = f"initial run: previous {initial_lookback_days} days"
    else:
        overlap_start = last_successful_end.astimezone(tz) - timedelta(days=overlap_days)
        start = min(week_start_dt, overlap_start)
        reason = f"last successful window end {last_successful_end.astimezone(tz):%Y-%m-%d %H:%M} − {overlap_days} days overlap"
        if overlap_start < week_start_dt - timedelta(days=overlap_days):
            reason += " (gap since last success is being back-filled)"
    return Window(start=start, end=end, research_week_start=week_start, research_week_end=week_end, reason=reason)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
