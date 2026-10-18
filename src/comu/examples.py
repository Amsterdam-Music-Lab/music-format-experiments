import comu
import pandas as pd
import pitchtypes as pt
import importlib
import matplotlib.pyplot as plt

importlib.reload(comu.interval_relations)
importlib.reload(comu)


def example_chord_labels(include: comu.IntervalRelations = "intersects", clip=False):
    """
    In a score with notes and chord labels,
    return groups of notes under a chord label
    """
    # create some notes
    notes = comu.SpanEvents(
        pd.DataFrame(
            {
                "span": pd.arrays.IntervalArray.from_breaks(
                    [0, 1.5, 2.5, 3, 4, 5], closed="left"
                ),
                "pitch": [pt.SpelledPitch(p) for p in ["C4", "E4", "D4", "B3", "C4"]],
            }
        )
    )
    # print("Notes:", notes)

    # create some chord labels
    chords = comu.EventPartition(
        pd.DataFrame({"onset": [0, 2, 4], "label": ["C", "G7", "C"]})
    )  # .spans(5)
    # print("Chord labels:", chords)

    # create and plot the whole score
    score = comu.Score({"notes": notes, "labels": chords})
    print(score)
    fig, ax = plt.subplots(4, 1, sharex=True)
    score.plot(ax[0])

    # merge notes with labels by interval relation
    notes_with_labels = score.merge("notes", "labels", by=include)
    # clip notes (optional)
    if clip:
        notes_with_labels = notes_with_labels.clip("span_labels")
    print(notes_with_labels)

    # Group merged events by label (via the onset column).
    # For visualization, create a new score for each group.
    for i, (onset, group) in enumerate(notes_with_labels.events.groupby(["onset"])):
        gscore = comu.Score()
        gscore["notes"] = group.as_spans
        label_spans = score["labels"].spans(score.end).events
        gscore["labels"] = label_spans[label_spans["onset"] == onset].as_spans
        print(gscore)
        gscore.plot(ax[i + 1])
    fig.show()


def example_relative_pitch():
    """
    In a score with notes and keys, express the pitch of notes relative to the local key.
    """
    # create some notes
    notes = comu.SpanEvents(
        pd.DataFrame(
            {
                "span": pd.arrays.IntervalArray.from_breaks(
                    [0, 1.5, 2.5, 3, 4, 5], closed="left"
                ),
                "pitch": [pt.SpelledPitch(p) for p in ["C4", "E4", "D4", "B3", "C4"]],
            }
        )
    )

    # create some keys
    keys = comu.EventPartition(
        pd.DataFrame(
            {
                "onset": [0, 2, 4],
                "root": [pt.SpelledPitchClass(p) for p in ["C", "G", "C"]],
            }
        )
    )

    # create and plot a score
    score = comu.Score({"notes": notes, "keys": keys})
    print(score)
    fig, ax = plt.subplots(2, 1, sharex=True)
    score.plot(ax[0])

    # merge notes with keys, convert pitch to relative
    rel_notes = score.merge("notes", "keys", by="starts_in").events
    rel_notes["pitch"] = rel_notes.apply(
        lambda row: row["pitch"].to_class() - row["root"], axis=1
    )

    # add relative notes to score and plot
    score["notes_relative"] = rel_notes.as_spans
    score.plot(ax[1])

    fig.show()
