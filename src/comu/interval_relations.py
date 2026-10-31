import functools
import operator
import typing
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas_interval_types import Interval, IntervalArray
from pandas_interval_types.interval_relations import (
    contains,
    contains_end,
    contains_start,
    contains_strict,
    ends_with,
    equal,
    finishes,
    intersection,  # noqa: F401
    intersects,
    meets,
    overlaps,
    point_after_interval,
    point_before_interval,
    point_ends_interval,
    point_in_interval,
    point_starts_interval,
    point_strictly_inside_interval,
    precedes,
    starts,
    starts_with,
)

# Interval Relations
# ==================

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
    "starts_with",  # = beginnings coincide (starts v equal v is_started)
    "ends_with",  # = ends coincide (ends v equal v is_ended)
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
    ("starts_with", "starts_with"),
    ("ends_with", "ends_with"),
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
        case "starts_with":
            return starts_with(a, b)
        case "ends_with":
            return ends_with(a, b)
        # list of relations:
        case list() as relations:
            return functools.reduce(
                operator.or_,
                (relates(a, b, rel) for rel in relations),
                False,
            )
        case _ as unreachable:
            typing.assert_never(unreachable)


# Point-Interval Relations
# ========================

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
                (point_relates(point, interval, rel) for rel in relations),
                False,
            )
        case _ as unreachable:
            typing.assert_never(unreachable)


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
    if left in df2.columns:
        left = left + (suffixes[0] or "")
    if right in df1.columns:
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
