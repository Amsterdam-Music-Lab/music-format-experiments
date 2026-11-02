import io
import re
import tomllib
from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal, NamedTuple

import pandas as pd
import pitchtypes as pt
from pandas_fractions import FractionArray
from pandas_interval_types import IntervalArray

import comu


class SpineType(NamedTuple):
    event_type: Literal["points", "spans", "states"] | None
    parse: Callable[[str], Any] | None


def parse_key(key):
    root, mode = key.split("-")
    return {"root": pt.IntervalClass(root), "mode": mode}


TYPE_PARSERS = {
    "pitch": SpineType("spans", lambda p: {"pitch": pt.SpelledPitch(p)}),
    "pitch_class": SpineType("spans", lambda p: {"pitch": pt.SpelledPitchClass(p)}),
    "interval": SpineType("spans", lambda i: {"interval": pt.SpelledInterval(i)}),
    "interval_class": SpineType(
        "spans", lambda i: {"interval": pt.SpelledIntervalClass(i)}
    ),
    "key": SpineType("states", parse_key),
    "meter": SpineType("states", None),
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
    for spine_name in df.columns:
        if spine_name == "time":
            explicit_time = parse_time_column(df["time"], header["time"]["type"])
        elif "(" not in spine_name:
            spine_parts = df[spine_name].str.extract(r"((?P<dur>[^:]*):)?(?P<data>.*)")
            df_data[spine_name] = spine_parts["data"]

            durations_str = spine_parts["dur"].replace("", None)
            durations = parse_time_column(durations_str, header["time"]["type"])
            spine_type = TYPE_PARSERS.get(
                header["spines"].get(spine_name, {}).get("type")
            )
            event_type = (
                TYPE_PARSERS[spine_type].event_type
                if spine_type in TYPE_PARSERS
                else None
            )
            if (not durations.isna().all()) or (event_type == "spans"):
                df_durations[spine_name] = durations
        else:
            df_data[spine_name] = df[spine_name]

    # determine onsets:
    record_durations = (
        explicit_time if explicit_time is not None else df_durations.min(axis=1)
    )
    onsets = record_durations.cumsum() - record_durations

    # collect primary columns as event series
    primary_spines = [
        name for name in df.columns if ("(" not in name) and (name != "time")
    ]
    score = comu.Score(header=header)

    for spine in primary_spines:
        # create dataframe with non-null tokens
        data = df_data[spine]
        mask = data != "."
        event_df = pd.DataFrame({"token": data[mask]})
        event_type = None  # the annotated event type in the metadata

        # if type is known:
        # - if has parser: parse tokens and add new columns
        # - otherwise: copy "tokens" to column with type name
        if "type" in header["spines"].get(spine, {}):
            # try to parse
            spine_type = header["spines"][spine]["type"]
            if spine_type in TYPE_PARSERS:
                event_type = TYPE_PARSERS[spine_type].event_type
                parser = TYPE_PARSERS[spine_type].parse
                if parser:
                    parsed = event_df["token"].apply(lambda a: pd.Series(parser(a)))
                    event_df = pd.concat([event_df, parsed], axis=1)
                else:
                    event_df[spine_type] = event_df["token"]
            else:
                event_df[spine_type] = event_df["token"]

            # drop tokens column?
            if not keep_tokens:
                event_df = event_df.drop("token", axis=1)

        # add secondary spines as local columns
        for other_spine in [c for c in df.columns if f"({spine})" in c]:
            sub_col_name = other_spine.split("(")[0]
            event_df[sub_col_name] = df_data[other_spine][mask]

        # add time and create events object
        if (event_type == "spans") or (
            (not event_type) and (spine in df_durations.columns)
        ):
            # has duration or marked as spans: span events
            offsets = onsets[mask] + df_durations[spine][mask]
            spans = pd.Series(
                IntervalArray.from_arrays(onsets[mask], offsets), onsets[mask].index
            )
            event_df.insert(0, "span", spans)
            score[spine] = comu.SpanEvents(event_df.reset_index(drop=True))
        else:
            # has no duration: point or state events
            event_df.insert(0, "onset", onsets[mask])
            if event_type == "states":
                score[spine] = comu.StateEvents(event_df.reset_index(drop=True))
            else:
                score[spine] = comu.PointEvents(event_df.reset_index(drop=True))

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
    df = pd.read_table(buf, dtype=str)
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
