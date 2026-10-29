import pandas as pd
import numpy as np
import numpy.typing as npt
import pandas_interval_types as pi
from pandas_interval_types import Interval, IntervalArray
from .interval_relations import (
    IntervalRelations,
    PointIntervalRelations,
    relates,
    point_relates,
    intersection,
    merge_on_intervals,
    merge_intervals_points,
    merge_points_intervals,
    merge_by_relation,
)
import matplotlib.pyplot as plt
import typing
from typing import Optional, Any, Callable
import operator
from collections import UserDict


@pd.api.extensions.register_dataframe_accessor("as_points")
class PointEvents:
    """
    A set of point events.

    Events are stored in the `events` attribute
    as a dataframe containing at least an `onset` column.
    """

    name: str | None
    events: pd.DataFrame

    def __init__(self, events: pd.DataFrame, name: str | None = None):
        """Create PointEvents from a DataFrame."""
        if "onset" not in events.columns:
            raise AttributeError("Must have 'onset' column.")
        self.events = events
        self.name = name

    def __repr__(self):
        """Print PointEvents."""
        return f"<PointEvents> {self.name or ''}\n" + repr(self.events)

    @property
    def end(self):
        return self.events["onset"].max()

    def add_to_plot(self, ax, label):
        """Add PointEvents to an axis."""
        pass  # TODO

    def map_time(self, f: Callable) -> "PointEvents":
        """
        Maps a function over the time axis of these events.

        The function must take and return an array of time points.
        """
        new_onsets = f(self.events["onset"])
        new_events = self.events.assign(onset=new_onsets)
        return PointEvents(new_events)

    def in_interval(
        self,
        interval: Interval | IntervalArray,
        by: PointIntervalRelations = "in",
    ) -> bool | npt.NDArray[np.bool]:
        """
        Select all points in a given interval.

        Returns a boolean mask.
        """
        return point_relates(self.events["onset"], interval, by)

    def merge_with_points(
        self,
        right_events: "PointEvents",
        right_label: str | None = None,
        by: Callable = operator.__eq__,
    ) -> "PointEvents":
        """merge with intersecting point events."""
        right_label = right_label or right_events.name or "right"
        merged = merge_by_relation(
            self.events,
            right_events.events,
            "onset",
            "onset",
            by=by,
            suffixes=(None, "_" + right_label),
        )
        return PointEvents(merged)

    def merge_with_spans(
        self,
        right_events: "SpanEvents",
        right_label: str | None = None,
        by: PointIntervalRelations = "in",
    ) -> "PointEvents":
        right_label = right_label or right_events.name or "right"
        merged = merge_points_intervals(
            self.events,
            right_events.events,
            "onset",
            "span",
            by=by,
            suffixes=(None, "_" + right_label),
        )
        return PointEvents(merged)


@pd.api.extensions.register_dataframe_accessor("as_spans")
class SpanEvents:
    """
    A set of span events.

    Events are stored in the `events` attribute
    as a dataframe containing at least a `span` column that contains intervals.
    """

    name: str | None
    events: pd.DataFrame

    def __init__(self, events: pd.DataFrame, name: str | None = None):
        # ensure that the span column is an IntervalArray with right-open intervals
        if "span" not in events.columns:
            raise AttributeError("Must have 'span' column.")
        self.events = events
        self.name = name

    def __repr__(self):
        return f"<SpanEvents> {self.name or ''}\n" + repr(self.events)

    @property
    def end(self):
        return self.events["span"].array.right.max()

    def add_to_plot(self, ax: plt.Axes, label: str | None):
        """
        Add these events to a plot.
        """
        label = label or self.name
        segments = self.events["span"].array
        color = ax._get_lines.get_next_color()
        ax.broken_barh(
            [(seg.left, seg.right - seg.left) for seg in segments],
            (label, 0.8),
            align="center",
            color=color,
            edgecolor=color,
            alpha=0.7,
            lw=2,
        )
        for _, row in self.events.iterrows():
            ax.text(row["span"].left, label, f" {row.iloc[1]}", va="center")

    def map_time(self, f: Callable) -> "SpanEvents":
        """
        Maps a function over the time axis of these events.

        The function must take and return an array of time points.
        """
        new_spans = self.events["span"].array.map_time(f)
        new_events = self.events.assign(span=new_spans)
        return SpanEvents(new_events)

    def clip(self, to: pi.Interval | pi.IntervalArray | str):
        """Return the intervals in this collection clipped to another interval."""
        if isinstance(to, str):
            interval = self.events[to].array
        else:
            interval = to

        clipped_spans = intersection(self.events["span"].array, interval)
        events = self.events.assign(span=clipped_spans)

        return events.as_spans

    def in_interval(
        self,
        interval: pi.Interval | pi.IntervalArray,
        by: IntervalRelations = "intersects",
    ):
        """
        Selects all spans in a given interval.
        Returns a boolean mask.
        """
        return relates(interval, self.events["span"].array, relation=by)

    def merge_with_points(
        self,
        right_events: PointEvents,
        right_label: str | None = None,
        by: PointIntervalRelations = "in",
    ) -> "SpanEvents":
        right_label = right_label or right_events.name or "right"
        merged = merge_intervals_points(
            self.events,
            right_events.events,
            "span",
            "onset",
            by=by,
            suffixes=(None, "_" + right_label),
        )
        return SpanEvents(merged)

    def merge_with_spans(
        self,
        right_events: "SpanEvents",
        right_label: str | None = None,
        by: IntervalRelations = "intersects",
    ) -> "SpanEvents":
        right_label = right_label or right_events.name or "right"
        merged = merge_on_intervals(
            self.events,
            right_events.events,
            "span",
            "span",
            by=by,
            suffixes=(None, "_" + right_label),
        )
        return SpanEvents(merged)


@pd.api.extensions.register_dataframe_accessor("as_states")
class StateEvents:
    """
    A series of events representing state changes.

    Events are stored in the `events` attribute
    as a dataframe containing at least an `onset` column.
    Events are interpreted to last from their own onset until the next onset.
    """

    name: str | None
    _events: pd.DataFrame

    def __init__(self, events: pd.DataFrame, name: str | None = None):
        if "onset" not in events.columns:
            raise AttributeError("Must have 'onset' column.")
        self._events = events.sort_values("onset")
        self.name = name

    def __repr__(self):
        return f"<StateEvents> {self.name or ''}\n" + repr(self.events)

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events):
        self._events = events.sort_values("onset")

    @property
    def end(self):
        return self.events["onset"].max()

    def add_to_plot(self, ax, label):
        onsets = self.events["onset"]
        color = ax._get_lines.get_next_color()
        # ax.scatter(onsets, np.full(onsets.shape, label), color=color, marker="|")
        ax.broken_barh(
            [(onset, 0) for onset in onsets],
            (label, 0.8),
            align="center",
            edgecolor=color,
            alpha=0.7,
            lw=2,
        )
        for _, row in self.events.iterrows():
            ax.text(row["onset"], label, f" {row.iloc[1]}", va="center")

    def onsets(self):
        """Return a `PointEvents` that contains the beginning of each state."""
        return PointEvents(self.events)

    def spans(self, end=np.inf):
        """
        Return a `SpanEvents` with active time span of each state as a span event.
        The endpoint of the last state is `np.inf` by default but can be changed.
        """
        onsets = self._events["onset"].values
        offsets = np.append(onsets[1:], end)
        spans = IntervalArray.from_arrays(onsets, offsets)
        span_events = self._events.assign(span=spans)  # .drop("onset", axis=1)
        return SpanEvents(span_events, self.name)

    def map_time(self, f: Callable) -> "StateEvents":
        """
        Maps a function over the time axis of these events.

        The function must take and return an array of time points.
        """
        new_onsets = f(self.events["onset"])
        new_events = self.events.assign(onset=new_onsets)
        return StateEvents(new_events, self.name)


Events = PointEvents | SpanEvents | StateEvents


E = typing.TypeVar("E", bound=Events)


def _set_name(events: E, name: str) -> E:
    """
    Sets the `name` attribute for an event series.

    Returns a new event series of the same time, not modifying the input.
    """
    return type(events)(events.events, name)


class Score(UserDict[str, Events]):
    """
    An extremely simplified representation of a score.

    Each type of events is represented by an event dataframe.
    """

    header: dict[str, Any]

    def __init__(
        self,
        init: Optional[dict[str, Events]] = None,
        header: dict[str, Any] = {},
        **kwargs: Events,
    ):
        if init:
            super().__init__(init, **kwargs)
        else:
            super().__init__(**kwargs)
        self.header = header

    def __setitem__(self, key, value):
        if isinstance(value, Events):
            value = _set_name(value, key)
            super().__setitem__(key, value)
        else:
            raise TypeError(f"A score cannot store spines of type {type(value)}")

    def __repr__(self):
        """Print a score."""
        parts = ["-" + key + ": " + repr(value) for key, value in self.items()]
        return "Score:\n" + "\n".join(parts)

    def plot(self, ax=None, **fig_args):
        """Plot a score."""
        if ax is None:
            if "figsize" not in fig_args:
                fig_args["figsize"] = (6.4, 0.6 * len(self))
            fig, ax = plt.subplots(**fig_args)
        else:
            fig = None

        for label, events in self.items():
            events.add_to_plot(ax, label)
        return fig

    @property
    def end(self):
        latest = [events.end for events in self.values()]
        return max(latest)

    def map_time(self, f: Callable) -> "Score":
        """
        Maps a function over the time axis of these events.

        The function must take and return an array of time points.
        """
        return Score({name: events.map_time(f) for name, events in self.items()})

    # TODO:
    # implement a general merge method as a more convenient merging interface
    # - dispatches on spine types
    # - knows spine names and thus `right_label`
    # - type-dependent default relations
    # - conversion from states to spans/points
