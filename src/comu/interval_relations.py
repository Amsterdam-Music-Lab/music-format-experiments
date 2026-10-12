import pandas as pd
import numpy.typing as npt
import numpy as np
import typing
from typing import Optional


def contains(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    # we have to compute these manually because
    # IntervalArray does not support the corresponding attributes
    a_closed_left = a.closed == "left" or a.closed == "both"
    b_open_left = b.closed == "right" or b.closed == "neither"
    a_closed_right = a.closed == "right" or a.closed == "both"
    b_open_right = b.closed == "left" or b.closed == "neither"

    # check left boundary
    if a_closed_left or b_open_left:
        fits_left = a.left <= b.left
    else:
        fits_left = a.left < b.left

    # check right boundary
    if a_closed_right or b_open_right:
        fits_right = a.right >= b.right
    else:
        fits_right = a.right > b.right

    return fits_left & fits_right


def contains_beginning(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    a_closed_left = a.closed == "left" or a.closed == "both"
    a_open_right = a.closed == "left" or a.closed == "neither"
    b_open_left = b.closed == "right" or b.closed == "neither"

    # check left boundary
    if a_closed_left or b_open_left:
        fits_left = a.left <= b.left  # a[b[ a[b( a(b(
    else:
        fits_left = a.left < b.left  # a(b[

    # check right boundary
    if a_open_right or b_open_left:
        fits_right = a.right > b.left  # b[a) b(a) b(a]
    else:
        fits_right = a.right >= b.left  # b[a]

    return fits_left & fits_right


def contains_end(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    a_open_left = a.closed == "right" or a.closed == "neither"
    a_closed_right = a.closed == "right" or a.closed == "both"
    b_open_right = b.closed == "left" or b.closed == "neither"

    # check left boundary
    if a_open_left or b_open_right:
        fits_left = a.left < b.right  # a(b] a(b) a[b)
    else:
        fits_left = a.left <= b.right  # a[b]

    # check right boundary
    if a_closed_right or b_open_right:
        fits_right = a.right >= b.right  # b]a] b)a] b)a)
    else:
        fits_right = a.right > b.right  # b]a)

    return fits_left & fits_right


def overlaps(
    a: pd.Interval | pd.arrays.IntervalArray, b: pd.Interval | pd.arrays.IntervalArray
) -> bool | npt.NDArray[np.bool]:
    # (a.right >= b.left) and (b.right >= a.left)
    a_open_left = a.closed == "right" or a.closed == "neither"
    a_open_right = a.closed == "left" or a.closed == "neither"
    b_open_left = b.closed == "right" or b.closed == "neither"
    b_open_right = b.closed == "left" or b.closed == "neither"

    if a_open_right or b_open_left:
        fits_left = a.right > b.left
    else:
        fits_left = a.right >= b.left

    if a_open_left or b_open_right:
        fits_right = b.right > a.left
    else:
        fits_right = b.right >= a.left

    return fits_left & fits_right


IntervalRelation = typing.Literal[
    "overlaps", "contains_beginning", "contains_end", "contains"
]


def relates(
    a: pd.Interval | pd.arrays.IntervalArray,
    b: pd.Interval | pd.arrays.IntervalArray,
    relation: IntervalRelation = "overlaps",
) -> bool | npt.NDArray[np.bool]:
    """
    Checks whether interval a is related to interval b by 'relation'.
    a or b can be an IntervalArray.
    Returns a bool or bool array.
    """
    match relation:
        case "overlaps":
            return overlaps(a, b)
        case "contains_beginning":
            return contains_beginning(a, b)
        case "contains_end":
            return contains_end(a, b)
        case "contains":
            return contains(a, b)
        case _ as unreachable:
            typing.assert_never(unreachable)


def smallest_common_boundaries(
    a: pd.IntervalClosedType, b: pd.IntervalClosedType
) -> pd.IntervalClosedType:
    if a == b:
        return a
    elif a == "both":
        return b
    elif b == "both":
        return a
    else:
        return "neither"


def intersection(
    a: pd.Interval | pd.arrays.IntervalArray,
    b: pd.Interval,
    closed: Optional[pd.IntervalClosedType] = None,
) -> pd.Interval | pd.arrays.IntervalArray:
    if isinstance(a, pd.Interval):
        return pd.Interval(
            max(a.left, b.left),
            min(a.right, b.right),
            closed=closed or smallest_common_boundaries(a.closed, b.closed),
        )
    elif isinstance(a, pd.arrays.IntervalArray):
        return pd.arrays.IntervalArray.from_arrays(
            np.maximum(a.left, b.left),
            np.minimum(a.right, b.right),
            closed=closed or smallest_common_boundaries(a.closed, b.closed),
        )
    else:
        typing.assert_never(a)
