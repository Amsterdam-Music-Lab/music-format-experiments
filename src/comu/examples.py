import comu
import pandas as pd
import pitchtypes as pt
import importlib
import matplotlib.pyplot as plt

importlib.reload(comu.interval_relations)
importlib.reload(comu)


def example_chord_labels(include: comu.IntervalRelation = "overlaps", clip=False):
    """
    In a score with notes and chord labels,
    return groups of notes under a chord label
    """
    notes = comu.SpanEvents(pd.DataFrame({
        "span": pd.arrays.IntervalArray.from_breaks([0, 1.5, 2.5, 3, 4, 5],
                                                    closed="left"),
        "pitch": [pt.SpelledPitch(p) for p in ["C4", "E4", "D4", "B3", "C4"]]
    }))
    # print("Notes:", notes)

    chords = comu.SpanEvents(pd.DataFrame({
        "span": pd.arrays.IntervalArray.from_breaks([0, 2, 4, 5],
                                                    closed="left"),
        "label": ["C", "G7", "C"]
    }))
    # print("Chord labels:", chords)

    score = comu.Score({
        "notes": notes,
        "labels": chords
    })
    print(score)
    fig, ax = plt.subplots(4, 1, sharex=True)
    score.plot(ax[0])

    groups = score.group_by_events("notes", "labels", include=include, clip=clip)
    for i, group in enumerate(groups):
        print("\n== Group ==")
        print(group)
        group.plot(ax[i+1])
    fig.show()
