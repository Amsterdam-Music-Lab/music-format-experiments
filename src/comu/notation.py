import pandas as pd
from pandas_fractions import FractionArray
import numpy as np
import comu
import io
import tomllib
import pitchtypes as pt
from typing import Any
import re
from fractions import Fraction


def parse_key(key):
    root, mode = key.split("-")
    return {"root": pt.IntervalClass(root), "mode": mode}


TYPE_PARSERS = {
    "pitch": lambda p: {"pitch": pt.SpelledPitch(p)},
    "pitch_class": lambda p: {"pitch": pt.SpelledPitchClass(p)},
    "interval": lambda i: {"interval": pt.SpelledInterval(i)},
    "interval_class": lambda i: {"interval": pt.SpelledIntervalClass(i)},
    "key": parse_key,
}


def make_header(string: str | None) -> dict[str, Any]:
    if string:
        header = tomllib.loads(string)
    else:
        header = {}

    # TODO: this could be a specialized data structure
    if "meta" not in header:
        header["meta"] = {}
    if "global" not in header:
        header["global"] = {}
    if "spines" not in header:
        header["spines"] = {}
    if "time" not in header:
        header["time"] = {}
    if "type" not in header["time"]:
        header["time"]["type"] = "recip"
    return header


RECIP_REGEX = re.compile(r"^((?P<num>\d+)/)?(?P<denom>\d+)?(?P<dots>\.*)$")


def parse_recip(string: str) -> Fraction:
    """
    Parses a time value in reciprocal notation.

    This notation is based on the humdrum **recip notation
    but extends to arbitrary reciprocal fractions.

        `(<num>/)?(<denom>)?\\.*`

    "x" => 1/x
    "x/y" => x/y
    "x/" => x
    "n." => n * 3/2

    Examples:
        parse_recip("4") -  > 0.25  # quarter note
        parse_recip("2.")  -> 0.75  # dotted half note
        parse_recip("3")   -> 0.33  # half-note triplet (1/3 of a whole note)
        parse_recip("2/.") -> 3     # dotted brevis
        parse_recip("3/2") -> 1.5   # dotted half note, equivalent to "2."
    """
    m = RECIP_REGEX.match(string)
    if m:
        print(m.groupdict())
        num = int(m.group("num") or 1)
        denom = int(m.group("denom") or 1)
        dots = len(m.group("dots") or "")
    else:
        raise ValueError(f"'{string}' is not a valid time value.")
    return Fraction(num, denom) * (2 - Fraction(1, 2**dots))


def parse_recip_column(col) -> FractionArray:
    """Like parse_recip but applied to an entire column."""
    parts = col.str.extract(RECIP_REGEX)
    num = parts["num"].fillna("1").astype("int")
    denom = parts["denom"].fillna("1").astype("int")
    dots = parts["dots"].str.len()
    base = FractionArray(num, denom)
    dot_factor = 2 - Fraction(1, 2) ** dots
    return base * dot_factor


def parse_time_column(col: pd.Series, time_type: str):
    match time_type:
        case "recip":
            return parse_recip_column(col)
        case "numeric":
            return pd.to_numeric(col)
        case _:
            raise ValueError(f"Unknown time type '{time_type}'.")


def df_to_score(
    df: pd.DataFrame, header: dict[str, Any], keep_tokens: bool
) -> comu.Score:
    # parse time
    df_durations = pd.DataFrame()
    df_data = pd.DataFrame()
    explicit_time: pd.Series | None = None
    for column in df.columns:
        if column == "time":
            explicit_time = parse_time_column(df["time"], header["time"]["type"])
        elif isinstance(df[column].dtype, pd.StringDtype) and ("(" not in column):
            col_parts = df[column].str.extract(r"((?P<time>[^:]*):)?(?P<data>.*)")
            df_data[column] = col_parts["data"]

            durations = parse_time_column(col_parts["time"], header["time"]["type"])
            if not durations.isna().all():
                df_durations[column] = durations
        else:
            df_data[column] = df[column]

    # determine onsets:
    record_durations = (
        explicit_time if explicit_time is not None else df_durations.min(axis=1)
    )
    onsets = record_durations.cumsum() - record_durations

    # collect primary columns as event series
    primary_columns = [c for c in df.columns if ("(" not in c) and (c != "time")]
    score = comu.Score(header=header)

    for col in primary_columns:
        data = df_data[col]
        mask = data != "."
        event_df = pd.DataFrame({"token": data[mask]})

        # if type is known, parse tokens
        if col in header["spines"] and "type" in header["spines"][col]:
            # try to parse
            col_type = header["spines"][col]["type"]
            if col_type in TYPE_PARSERS:
                parser = TYPE_PARSERS[col_type]
                parsed = event_df["token"].apply(lambda a: pd.Series(parser(a)))
                event_df = pd.concat([event_df, parsed], axis=1)
            else:
                event_df[col_type] = event_df["token"]

            # drop tokens column?
            if not keep_tokens:
                event_df = event_df.drop("token", axis=1)

        # add secondary columns
        for other_col in [c for c in df.columns if f"({col})" in c]:
            sub_col = other_col.split("(")[0]
            event_df[sub_col] = df_data[other_col][mask]

        # add time and create events object
        if col in df_durations.columns:
            # has duration: span events
            offsets = onsets[mask] + df_durations[col][mask]
            event_df.insert(
                0,
                "span",
                pd.Series(
                    pd.arrays.IntervalArray.from_arrays(
                        onsets[mask].astype("float"),
                        offsets.astype("float"),
                        closed="left",
                    ),
                    onsets[mask].index,
                ),
            )
            score[col] = comu.SpanEvents(event_df.reset_index(drop=True))
        else:
            # has no duration: point events
            event_df.insert(0, "onset", onsets[mask])
            score[col] = comu.PointEvents(event_df.reset_index(drop=True))

    return score


def read_comu(input_str, keep_tokens=True) -> comu.Score:
    """
    Read a string in comu notation.

    This is a very rough sketch of a possible notation.
    """

    # parse header
    buf = io.StringIO(input_str.strip())
    line = buf.readline()
    if line.startswith("---"):
        header_lines = []
        while not (line := buf.readline()).startswith("---"):
            header_lines.append(line)
        header = make_header("\n".join(header_lines))
    else:
        header = make_header(None)
        buf.seek(0)

    # read table
    df = pd.read_table(buf, dtype={"time": str})
    return df_to_score(df, header, keep_tokens=keep_tokens)


EXAMPLE = """
---
[spines.notes]
type = "pitch"
[spines.sections]
type = "section"
---
notes	id(notes)	sections
4:.	.	2:A
8:C4	n1	.
4.:D4	n2	.
4:.	.	2:volta1
8:B3	n3	.
4.:C4	n4	.
4:.	.	1.:volta2
4:B3	n5	.
2:C4	n6	.
"""

# This example has an explicit time column.
# Note how the first row has time 8 (= 1/8)
# but the first note has duration 4 (= 1/4).
# This means that the second note (duration 4. = 3/8)
# overlaps with the first note by 1/8.
EXAMPLE_EXPLICIT_TIME = """
---
spines.notes.type = "pitch"
---
time	notes	mark
8	4:C4	A
4.	4.:C4	.
4	4:C4	B
"""


def get_example_score() -> comu.Score:
    return read_comu(EXAMPLE)
