import pandas as pd
import comu
import io
import tomllib
import pitchtypes as pt


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


def read_comu(input_str, keep_tokens=True) -> comu.Score:
    """
    Read a string in comu notation.

    This is a very rough sketch of a possible notation.
    """

    # parse header
    header = {}
    buf = io.StringIO(input_str.strip())
    line = buf.readline()
    if line.startswith("---"):
        header_lines = []
        while not (line := buf.readline()).startswith("---"):
            header_lines.append(line)
        header = tomllib.loads("\n".join(header_lines))
    else:
        # reset to beginning
        buf.seek(0)

    # TODO: this could be a specialized data structure
    if "meta" not in header:
        header["meta"] = {}
    if "global" not in header:
        header["global"] = {}
    if "spines" not in header:
        header["spines"] = {}

    # read table
    df = pd.read_table(buf)

    # parse time
    df_durations = pd.DataFrame()
    df_data = pd.DataFrame()
    for column in df.columns:
        if isinstance(df[column].dtype, pd.StringDtype) and ("(" not in column):
            col_parts = df[column].str.extract(r"((?P<time>[^:]*):)?(?P<data>.*)")
            df_data[column] = col_parts["data"]

            durations = pd.to_numeric(col_parts["time"])
            if not durations.isna().all():
                df_durations[column] = durations
        else:
            df_data[column] = df[column]

    # determine onsets:
    record_durations = df_durations.min(axis=1)
    onsets = record_durations.cumsum() - record_durations

    # collect primary columns as event series
    primary_columns = [c for c in df.columns if "(" not in c]
    events = comu.Score(header=header)

    for col in primary_columns:
        data = df_data[col]
        mask = data != "."
        event_df = pd.DataFrame({"token": data[mask]})

        # if type is known, parse tokens
        if col in header["spines"] and "type" in header["spines"][col]:
            col_type = header["spines"][col]["type"]
            if col_type in TYPE_PARSERS:
                parser = TYPE_PARSERS[col_type]
                parsed = event_df["token"].apply(lambda a: pd.Series(parser(a)))
                event_df = pd.concat([event_df, parsed], axis=1)
            else:
                event_df[col_type] = event_df["token"]
            if not keep_tokens:
                event_df = event_df.drop("token", axis=1)

        # add secondary columns
        for other_col in [c for c in df.columns if f"({col})" in c]:
            sub_col = other_col.split("(")[0]
            event_df[sub_col] = df_data[other_col][mask]

        # add time and create events object
        if col in df_durations.columns:
            # has duration: span events
            event_df.insert(
                0,
                "span",
                pd.Series(
                    pd.arrays.IntervalArray.from_arrays(
                        onsets[mask],
                        onsets[mask] + df_durations[col][mask],
                        closed="left",
                    ),
                    onsets[mask].index,
                ),
            )
            events[col] = comu.SpanEvents(event_df.reset_index(drop=True))
        else:
            # has no duration: point events
            event_df.insert(0, "onset", onsets[mask])
            events[col] = comu.PointEvents(event_df.reset_index(drop=True))

    return events


EXAMPLE = """
---
[spines.notes]
type = "pitch"
[spines.sections]
type = "section"
---
notes	id(notes)	sections
1:.	.	2:A
0.5:C4	n1	.
1.5:D4	n2	.
1:.	.	2:volta1
0.5:B3	n3	.
1.5:C4	n4	.
1:.	.	3:volta2
1:B3	n5	.
2:C4	n6	.
"""


def get_example_score() -> comu.Score:
    return read_comu(EXAMPLE)
