from collections import UserDict
import pandas as pd
import numpy as np
from .interval_relations import relates, IntervalRelation, intersection
import matplotlib.pyplot as plt


class Score(UserDict):
    """
    An extremely simplified representation of a score.

    Each type of events is represented by an event dataframe.
    """

    def __repr__(self):
        """Print a score."""
        parts = ["-" + key + ": " + repr(value) for key, value in self.data.items()]
        return "Score:\n" + "\n".join(parts)

    def plot(self, ax=None):
        """Plot a score."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        for label, events in self.items():
            events.add_to_plot(ax, label)
        return fig

    def group_by_events(
        self,
        events: str,
        grouper: str,
        include: IntervalRelation = "overlaps",
        clip: bool = False,
    ):
        """Group some events by other overlap with other events."""
        grouper_events = self[grouper]
        grouped_events = self[events]

        match grouper_events:
            case PointEvents():
                results = grouped_events.group_by_points(grouper_events)
            case SpanEvents():
                results = grouped_events.group_by_spans(
                    grouper_events, include=include, clip=clip
                )
            case EventPartition():
                results = grouped_events.group_by_spans(
                    grouper_events, include=include, clip=clip
                )

        def mkscore(item):
            label, group = item
            s = Score()
            s[events] = group
            s[grouper] = label
            return s

        return map(mkscore, results)


class PointEvents:
    """
    A set of point events.

    Events are stored in the `events` attribute
    as a dataframe containing at least an `onset` column.
    """

    events: pd.DataFrame

    def __init__(self, events: pd.DataFrame):
        """Create PointEvents from a DataFrame."""
        self.events = events

    def __repr__(self):
        """Print PointEvents."""
        return "<PointEvents>\n" + repr(self.events)

    def __getitem__(self, index):
        """Index PointEvents, returning a new PointEvents object."""
        item = self.events[index]
        match item:
            case pd.DataFrame():
                return PointEvents(item)
            case _:
                return item

    def add_to_plot(self, ax, label):
        """Add PointEvents to an axis."""
        pass  # TODO

    def in_interval(self, interval: pd.Interval):
        """
        Select all points in a given interval.

        Returns a boolean mask.
        """
        return self.events["onset"].isin(interval)  # TODO, this probably doesn't work

    def group_by_points(self, grouper_events):
        """Group by overlapping point events."""
        raise TypeError("Cannot group point events by point events.")

    def group_by_spans(
        self,
        grouper_events: "SpanEvents",
        include: IntervalRelation = "overlaps",
        clip: bool = False,
    ) -> list[tuple["SpanEvents", "PointEvents"]]:
        """
        Group points by spans.

        Returns a list of (span, points) pairs,
        where span is a row of the SpanEvents dataframe
        and points is a PointEvents object.
        """
        return [
            (SpanEvents(row.to_frame().T), self[self.in_interval(row["span"])])
            for _, row in grouper_events.events.iterrows()
        ]


class SpanEvents:
    """
    A set of span events.

    Events are stored in the `events` attribute
    as a dataframe containing at least a `span` column that contains intervals.
    """

    events: pd.DataFrame

    def __init__(self, events: pd.DataFrame):
        # ensure that the span column is an IntervalArray with right-open intervals
        events = events.assign(
            span=pd.arrays.IntervalArray(events["span"], closed="left")
        )
        self.events = events

    def __repr__(self):
        return "<SpanEvents>\n" + repr(self.events)

    def __getitem__(self, index):
        item = self.events[index]
        match item:
            case pd.DataFrame():
                return SpanEvents(item)
            case _:
                return item

    def add_to_plot(self, ax: plt.Axes, label: str):
        """
        Add these events to a plot.
        """
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

    def clip(self, interval: pd.Interval):
        """Return the intervals in this collection clipped to another interval."""
        print("clipping")
        clipped_spans = intersection(self.events["span"].array, interval, closed="left")
        events = self.events.assign(span=clipped_spans)
        return SpanEvents(events[~events["span"].array.is_empty])

    def in_interval(
        self, interval: pd.Interval, include: IntervalRelation = "overlaps"
    ):
        """
        Selects all spans in a given interval.
        Returns a boolean mask.
        """
        return relates(interval, self.events["span"].array, relation=include)

    def group_by_points(self, grouper_events: PointEvents):
        pass

    def group_by_spans(
        self,
        grouper_events: "SpanEvents",
        include: IntervalRelation = "overlaps",
        clip: bool = False,
    ) -> list[tuple["SpanEvents", "SpanEvents"]]:  # TODO: implement clipping
        """
        Groups points by spans.
        Returns a list of (span, points) pairs,
        where span is a row of the SpanEvents dataframe
        and points is a PointEvents object.
        """
        print(clip)
        return [
            (
                SpanEvents(row.to_frame().T),
                (
                    (span := row["span"])
                    and (result := self[self.in_interval(span, include=include)])
                    and (result.clip(span) if clip else result)
                ),
            )
            for _, row in grouper_events.events.iterrows()
        ]


class EventPartition:
    """
    An event partition.

    Events are stored in the `events` attribute
    as a dataframe containing at least an `onset` column.
    Events are interpreted to last from their own onset until the next onset.
    """

    def __init__(self, events: pd.DataFrame):
        self._events = events.sort_values("onset")

    def __repr__(self):
        return "<EventPartition>\n" + repr(self.events)

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events):
        self._events = events.sort_values("onset")

    def __getitem__(self, index):
        item = self.events[index]
        match item:
            case pd.DataFrame():
                return EventPartition(item)
            case _:
                return item

    def add_to_plot(self, ax, label):
        pass  # TODO

    def spans(self, end=np.inf):
        """
        Return a `SpanEvents` with the segments of the partition as span events.
        The endpoint of the last event is `np.inf` by default but can be changed.
        """
        onsets = self._events["onset"].values
        offsets = np.append(onsets[1:], end)
        spans = pd.arrays.IntervalArray.from_arrays(onsets, offsets, closed="left")
        span_events = self._events.assign(span=spans).drop("onset", axis=1)
        return SpanEvents(span_events)

    def group_by_points(self, grouper_events: PointEvents):
        pass

    def group_by_events(
        self,
        grouper_events: SpanEvents,
        include: IntervalRelation = "overlaps",
        clip: bool = False,
    ):
        pass
