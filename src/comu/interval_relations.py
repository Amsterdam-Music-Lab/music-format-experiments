import pandas as pd
import numpy.typing as npt
import numpy as np
import typing
from typing import Any
import operator
import functools

# Relation Functions
# ==================

# Allen relations
# ---------------


def precedes(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return a.right < b.left


def meets(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return a.right == b.left


def overlaps(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.left) & (a.right > b.left) & (a.right < b.right)


def starts(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left == b.left) & (a.right < b.right)


def contains_strict(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.left) & (a.right > b.right)


def finishes(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left > b.left) & (a.right == b.right)


def equal(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left == a.left) & (a.right == a.right) & (a.closed == b.closed)


# Extra relations
# ---------------


def contains(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    """constains_strict or is_started or is_finished"""
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left <= b.left) & (a.right >= b.right)


def contains_start(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left <= b.left) & (a.right > b.left)


def contains_end(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert a.closed == "left"
    assert b.closed == "left"
    return (a.left < b.right) & (a.right >= b.right)


def intersects(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
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
    a: pd.Interval | pd.arrays.IntervalArray,
    b: pd.Interval | pd.arrays.IntervalArray,
    relation: IntervalRelation | list[IntervalRelation],
) -> bool | npt.NDArray[np.bool]:
    """
    Checks whether interval a is related to interval b by relation.

    Parameters
    ----------
    a, b : Interval | IntervalArray
    relation : IntervalRelation | list[IntervalRelation]

    Returns
    -------
    bool | np.NDArray[np.bool]
      A boolean (array) indicating whether a and b are related by relation or not.
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


def intersection(
    a: pd.Interval | pd.arrays.IntervalArray,
    b: pd.Interval | pd.arrays.IntervalArray,
) -> pd.Interval | pd.arrays.IntervalArray:
    """Returns the intersection of intervals a and b."""
    assert a.closed == "left"
    assert b.closed == "left"
    if isinstance(a, pd.Interval):
        return pd.Interval(
            np.maximum(a.left, b.left), np.minimum(a.right, b.right), closed="left"
        )
    elif isinstance(a, pd.arrays.IntervalArray):
        return pd.arrays.IntervalArray.from_arrays(
            np.maximum(a.left, b.left), np.minimum(a.right, b.right), closed="left"
        )
    else:
        typing.assert_never(a)


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
    cross = pd.merge(
        df1,
        df2,
        # left_on=left,
        # right_on=right,
        how="cross",
        suffixes=suffixes,
        sort=sort,
    )
    if left == right:
        left = left + (suffixes[0] or "")
        right = right + (suffixes[1] or "")
    filtered = cross[relates(cross[left].array, cross[right].array, by)]

    return pd.DataFrame(filtered)


def in_interval(
    a: Any, i: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    assert i.closed == "left"
    return a >= i.left & a < i.right


# def is_overlapped(
#     a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
# ) -> bool | npt.NDArray[np.bool]:
#     a_open_left = a.closed == "right" or a.closed == "neither"
#     a_closed_right = a.closed == "right" or a.closed == "both"
#     b_open_right = b.closed == "left" or b.closed == "neither"

#     # check left boundary
#     if a_open_left or b_open_right:
#         fits_left = a.left < b.right  # a(b] a(b) a[b)
#     else:
#         fits_left = a.left <= b.right  # a[b]

#     # check right boundary
#     if a_closed_right or b_open_right:
#         fits_right = a.right >= b.right  # b]a] b)a] b)a)
#     else:
#         fits_right = a.right > b.right  # b]a)

#     return fits_left & fits_right
