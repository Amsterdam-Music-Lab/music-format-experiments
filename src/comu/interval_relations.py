import pandas as pd
from pandas import Interval
from pandas.arrays import IntervalArray
import numpy.typing as npt
import numpy as np
import typing
from typing import Any
import operator
import functools

# Interval Relations
# ==================

# Allen relations
# ---------------


def precedes(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return a.right < b.left


def meets(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return a.right == b.left


def overlaps(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.left) & (a.right > b.left) & (a.right < b.right)


def starts(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left == b.left) & (a.right < b.right)


def contains_strict(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.left) & (a.right > b.right)


def finishes(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left > b.left) & (a.right == b.right)


def equal(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left == a.left) & (a.right == a.right) & (a.closed == b.closed)


# Extra relations
# ---------------


def contains(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    """constains_strict or is_started or is_finished"""
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left <= b.left) & (a.right >= b.right)


def contains_start(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left <= b.left) & (a.right > b.left)


def contains_end(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.right) & (a.right >= b.right)


def intersects(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.right > b.left) & (b.right > a.left)


# Working with Relations
# ----------------------

IntervalRelation = typing.Literal[
    # Allen relations:
    "precedes",
    "follows",
    "meets",
    "is_met",
    "overlaps",
    "is_overlapped",
    "starts",
    "is_started",
    "contains_strict",
    "during_strict",
    "finishes",
    "is_finished",
    "equal",
    # extensions:
    "intersects",  # = any common time points
    "contains",  # = contains_strict or is_started or is_finished
    "during",  # = during_strict or starts or finishes
    "contains_start",
    "starts_in",
    "contains_end",
    "ends_in",
]

IntervalRelations = IntervalRelation | list[IntervalRelation]

CONVERSE_PAIRS: list[tuple[IntervalRelation, IntervalRelation]] = [
    # Allen:
    ("precedes", "follows"),
    ("meets", "is_met"),
    ("overlaps", "is_overlapped"),
    ("starts", "is_started"),
    ("contains_strict", "during_strict"),
    ("finishes", "is_finished"),
    ("equal", "equal"),
    # extensions:
    ("intersects", "intersects"),
    ("contains", "during"),
    ("contains_start", "starts_in"),
    ("ends_in", "contains_end"),
]
CONVERSES: dict[IntervalRelation, IntervalRelation] = {
    a: b for x, y in CONVERSE_PAIRS for a, b in [(x, y), (y, x)]
}


def converse(ir: IntervalRelations) -> IntervalRelations:
    """Returns the converse of a relation or set of relations."""
    match ir:
        case str():
            return CONVERSES[ir]
        case list():
            return [CONVERSES[rel] for rel in ir]


def relates(
    a: Interval | IntervalArray,
    b: Interval | IntervalArray,
    relation: IntervalRelation | list[IntervalRelation],
) -> bool | npt.NDArray[np.bool]:
    """
    Checks whether interval a is related to interval b by relation.

    Args:
      a, b: interval or interval array
      relation: the interval relation(s) to test between `a` and `b`.

    Returns:
      A boolean (array) indicating whether `a` and `b` are related by `relation`.
    """
    match relation:
        # Allen relations
        case "precedes":
            return precedes(a, b)
        case "follows":
            return precedes(b, a)
        case "meets":
            return meets(a, b)
        case "is_met":
            return meets(b, a)
        case "overlaps":
            return overlaps(a, b)
        case "is_overlapped":
            return overlaps(b, a)
        case "starts":
            return starts(a, b)
        case "is_started":
            return starts(b, a)
        case "contains_strict":
            return contains_strict(a, b)
        case "during_strict":
            return contains_strict(b, a)
        case "finishes":
            return finishes(a, b)
        case "is_finished":
            return finishes(b, a)
        case "equal":
            return equal(a, b)
        # extra relations:
        case "intersects":
            return intersects(a, b)
        case "contains":
            return contains(a, b)
        case "during":
            return contains(b, a)
        case "contains_start":
            return contains_start(a, b)
        case "starts_in":
            return contains_start(b, a)
        case "contains_end":
            return contains_end(a, b)
        case "ends_in":
            return contains_end(b, a)
        # list of relations:
        case list() as relations:
            return functools.reduce(
                operator.or_,
                map(lambda rel: relates(a, b, rel), relations),
                False,
            )
        case _ as unreachable:
            typing.assert_never(unreachable)


# Point-Interval Relations
# ========================

# Relation Functions
# ------------------


def point_before_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a < i.left


def point_starts_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a == i.left


def point_strictly_inside_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    """Returns True if the point is strictly within the interval."""
    assert i.closed == "left"
    return a > i.left & a < i.right


def point_ends_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a == i.right


def point_after_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a > i.right


def point_in_interval(
    a: Any, i: Interval | IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a >= i.left & a < i.right


# Working with point-interval relations
# -------------------------------------

PointIntervalRelation = typing.Literal[
    "before",
    "after",
    "starts",
    "ends",
    "in",
    "strictly_inside",
]

PointIntervalRelations = PointIntervalRelation | list[PointIntervalRelation]


def point_relates(
    point: Any,
    interval: Interval | IntervalArray,
    relation: PointIntervalRelation | list[PointIntervalRelation],
) -> bool | npt.NDArray[np.bool]:
    """Checks whether point is related to interval by relation."""
    match relation:
        case "before":
            return point_before_interval(point, interval)
        case "after":
            return point_after_interval(point, interval)
        case "starts":
            return point_starts_interval(point, interval)
        case "ends":
            return point_ends_interval(point, interval)
        case "in":
            return point_in_interval(point, interval)
        case "strictly_inside":
            return point_strictly_inside_interval(point, interval)
        # list of relations:
        case list() as relations:
            return functools.reduce(
                operator.or_,
                map(lambda rel: point_relates(point, interval, rel), relations),
                False,
            )
        case _ as unreachable:
            typing.assert_never(unreachable)


# Interval and Point Operations
# =============================


def intersection(
    a: Interval | IntervalArray, b: Interval | IntervalArray
) -> Interval | IntervalArray:
    """Returns the intersection of intervals a and b."""
    assert a.closed == "left"
    assert b.closed == "left"
    if isinstance(a, Interval):
        return Interval(
            np.maximum(a.left, b.left), np.minimum(a.right, b.right), closed="left"
        )
    elif isinstance(a, IntervalArray):
        return IntervalArray.from_arrays(
            np.maximum(a.left, b.left), np.minimum(a.right, b.right), closed="left"
        )
    else:
        typing.assert_never(a)


def map_interval(
    f: typing.Callable, interval: Interval | IntervalArray
) -> Interval | IntervalArray:
    """
    Maps a function over the start and end points of an interval (array).

    Args:
      f: A function that acts on the interval's `.left` and `.right` boundaries.
      interval: The interval that `f` should be applied to.

    Returns:
      A new interval (array) with updated boundaries.
    """
    assert interval.closed == "left"
    if isinstance(interval, Interval):
        return Interval(f(interval.left), f(interval.right), closed="left")
    elif isinstance(interval, IntervalArray):
        return IntervalArray.from_arrays(
            f(interval.left), f(interval.right), closed="left"
        )
    else:
        typing.assert_never(interval)


def translate(
    interval: Interval | IntervalArray, offset: Any
) -> Interval | IntervalArray:
    """Shift an interval by a given amount."""
    return map_interval(lambda t: t + offset, interval)


def scale(interval: Interval | IntervalArray, factor: Any) -> Interval | IntervalArray:
    """
    Scale the time axis of an interval.

    This simply multiplies the boundaries of `interval` by `factor`.
    """
    return map_interval(lambda t: t * factor, interval)


# Merge Functions
# ===============


def merge_by_relation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: str,
    right: str,
    by: typing.Callable,
    suffixes: tuple[str, str] | tuple[str, None] | tuple[None, str] = (
        "_x",
        "_y",
    ),
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge by some relation.

    Performs a cross merge (cartesian product) between df1 and df2.
    Filters the results by a custom relation `by`
    between the columns `left` from `df1` and `right` from `df2`.

    Args:
      df1: The left dataframe.
      df2: The right dataframe.
      left: The name of the merge column in `df1`.
      right: The name of the merge column in `df2`.
      by: A function that takes the two merge columns and returns a boolean column,
        indicating whether a row should be included in the merge or not.
      suffixies: a pair of `str | None`, indicating how columns in df1 or df2
        should be renamed in the case of conflicts.
      sort: Indicates whether rows should be sorted by join keys.

    Returns:
      A dataframe that merges columns from `df1` and `df2`
      for every pair of rows that satisfies `by`.
    """
    cross = pd.merge(df1, df2, how="cross", suffixes=suffixes, sort=sort)
    if left == right:
        left = left + (suffixes[0] or "")
        right = right + (suffixes[1] or "")
    filtered = cross[by(cross[left], cross[right])]
    return pd.DataFrame(filtered)


def merge_on_intervals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: str,
    right: str,
    by: IntervalRelations = "intersects",
    suffixes: tuple[str, str] | tuple[str, None] | tuple[None, str] = (
        "_x",
        "_y",
    ),
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge two dataframes on interval columns.

    The interval relation 'by' determines how the right events relate to the left.
    """

    def by_f(left_col, right_col):
        return relates(left_col.array, right_col.array, by)

    return merge_by_relation(df1, df2, left, right, by_f, suffixes, sort)


def merge_intervals_points(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: str,
    right: str,
    by: PointIntervalRelations = "in",
    suffixes: tuple[str, str] | tuple[str, None] | tuple[None, str] = (
        "_x",
        "_y",
    ),
    sort: bool = False,
) -> pd.DataFrame:
    """Merge two dataframes on intervals (left) and points (right)."""

    def by_f(left_col, right_col):
        return point_relates(right_col, left_col.array, by)

    return merge_by_relation(df1, df2, left, right, by_f, suffixes, sort)


def merge_points_intervals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: str,
    right: str,
    by: PointIntervalRelations = "in",
    suffixes: tuple[str, str] | tuple[str, None] | tuple[None, str] = (
        "_x",
        "_y",
    ),
    sort: bool = False,
) -> pd.DataFrame:
    """Merge two dataframes on points (left) and left (right)."""

    def by_f(left_col, right_col):
        return point_relates(left_col, right_col.array, by)

    return merge_by_relation(df1, df2, left, right, by_f, suffixes, sort)
