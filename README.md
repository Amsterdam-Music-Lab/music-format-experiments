# comu

This package implements some experiments in representing encoding symbolic music data for computational musicology.
See [examples.ipynb](examples.ipynb) for some examples.

## Design

This package explores two things:
- general representations (data structures) of scores that support useful operations
- a human-readable notation format for encoding scores into these structures.

The motivation for the first point is to define a sound formalism that works as a foundation for computational musicology packages, such as dimcat, CAMAT, or partitura, similar to how [relational algebra](https://en.wikipedia.org/wiki/Relational_algebra) is a common interface for database systems (SQL) and dataframe libraries (pandas, R dataframes, etc.).

### Data Structures

A piece of music is represented as a collection of events.
Events are organzied in series or "spines" (in analogy to humdrum, but without branching paths) and can be organized as point, span, or state events:

- Point events happen at a particular point but don't have a duration.
  Point events can mark particular moments in a piece or markers attached to a particular point of time.
- Span events have an onset and a duration (and hence an offset).
  Span events can generally overlap and have gaps within a spine.
  Spans encode entities with a temporal extension such as notes, harmonic labels, or formal segments.
- State events partition the piece into several non-overlapping states.
  Like spans, each segment in a partition has an onset and offset, but state events do not allow gaps or overlaps.
  Partitions can be conceputally reduced to point events by having the events mark the boundaries of a segment.
  They can be used to encode things that are changed at particular points and valid until the next change,
  such as time and key signatures.
  
These generic event types serve as containers for the actual contents of the score, each spine being restricted to a particular event type (e.g. notes, time signature, form markers etc.).
Taken together, the events and spines are similar to regions and tracks in a DAW or video editor.

Interval spans are generally considered to be closed on the left and open on the right, which means that the onset of an event is part of its span, but the offset is not.

#### Operations

Operations on these structures are mostly based on the temporal information of the events and their relations:

- interval relations based on something like [Allen's interval algebra](https://en.wikipedia.org/wiki/Allen's_interval_algebra)
  - analogous relations for points and between points and intervals
- taking the intersection of two intervals
- mapping over the time of events (e.g. shifting, scaling, converting to a different unit, quantizing etc.)
- "merging" (or "joining") series of events based on a relation (e.g., merge notes with the chord labels they intersect with).

These basic operations can be used to implement higher-level tasks.
Examples include:

- **Extracing notes under chord labels**:
  1. Find the on- and offset of each chord label.
  2. Choose an inclusion criterion (e.g. complete or partial overlap)
  3. Find notes that overlap with the label span according to (2).
  4. (optional) Clip notes that extend beyond the label span.
- **Expressing notes relative to local keys**:
  For each note:
  1. Find the local key by checking segments from key partition.
  2. Replace absolute pitch with interval to key root / scale degree.
- **Unfolding repetitions**: Given labelled region spans and a sequence of regions:
  1. For each region, identify overlapping events.
  2. Concatenate the regions by copying and shifting events:
     - points: copy and shift if within
     - spans: copy and shift if overlaps
     - partition: clip segment to region, copy and shift, merge *adjacent identical* segments
- **Inferring measures and metrical positions of notes**: given partition of time signatures
  1. For each segment in TS partition, create measure segments starting at beginning.
  2. For each note
     - find measure containing onset
     - determine relative position to that measure
     - e.g. determine metrical weight of position under given TS
- **Monophony check** for notes
  1. sort by onset
  2. check that offsets and consecutive onsets don't overlap
- **Quantization**
  1. define quantiztion function (time -> time or interval -> interval)
  2. map over all events
  - could map to different time axis (e.g. integer indices)
- **Slicing** the piece at note boundaries
  1. Collect all note changes (i.e., on- and offsets).
  2. Sort the break points and create corresponding slice spans.
  3. Select and clip notes overlapping with each slice.

Note that some of these tasks rely on content-type dependent semantics (e.g. time signatures: new segment = downbeat),
but can all be implemented in terms of generic event operations!

### Notation

This part explores possible notations for encoding the above structures in a human-readable but unambiguous and verifiable way.
The basic idea is similar to humdrum (tabular, spines, separation of inner and outer structure), except that there are common semantics for temporal relations that can be checked without having to know the contents of the events.
In addition, metadata and global data are handled in a systematic way using TOML (or possibly YAML).

Here is an example of a proposed notation.
Comments are marked by # and are not necessarily part of the file format (except the TOML segment).

```
--- # TOML header (like markdown with YAML header)
[meta]
title = "Piece 1"
composer = "me"
encoder = "me"

[global] # for values that hold throughout the piece
key = { type="key", value="G-minor" }

[spines.bar]
type = bar

[spines.vo1]
type = kern # could also be done in header? or designated type row?
staff = 0 # this is constant data for all events in the spine

[spines.vo2]
type = kern
staff = 1

[spines.id(vo1)]
type = id
to = vo1 # could be implied by name

[spines.id(vo2)]
type = id
to = vo2

[spines.timesig] # changes, so not in global, could also be spine-dependent
type = timesig

[spines.aux]
type = null
--- # end of header
bar	aux	vo2	id(vo2)	vo1	id(vo1)	timesig
=1	.	4:g2	v2n1	4:g4	v1n1	3/4
.	.	4:b2	v2n2	4:f#4	v1n2	.
.	.	4:c3	v2n3	8:.	.	.
.	.	.	.	4:e4	v1n3	.
=2	.	2:d3	v2n4	.	.	cad
.	.	.	.	8:d4	v1n4	.
.	.	.	.	8:f#4	v1n5	.
.	.	.	.	8:d4	v1n6	.
=3	0:.	4:g2	v2n5	4:g	v1n7	3/4
.	.	4:b2	v2n6	.	.	.
.	.	2:.	.	2:.	.	.
```

## Implementation

### Datastructures and Operations

The current prototype implementation is based on (pandas) dataframes.
Each spine is stored in a separate dataframe with columns depending on its event type (point, span, state) and content type (notes, chord labels, sections, keys etc.).
Spines of different event types are wrapped in classes (`PointEvents`, `SpanEvents` and `StateEvents`) that provide additional event-type specific methods.
Point and state event dataframe must have a column `onset` while span events must have a column `span` containing a pandas `IntervalArray` (left-closed).
The spines are collected in a `Score`, which is a dictionary mapping spine names to their event series and provides extra methods for plotting.

The dataframe representation allows the user to make use of the dataframe API, and event-specific operations are implement on top of these.
Interval, point, and point-interval relations are implemented as predicates on individual values or arrays (e.g. columns).
"Merges" are based on panda's "cross" merge (i.e., cartesian product) and filtering by a given relation.
This naive interpretation should be feasible for single pieces that are not too long, but might be a problem for very long pieces.

Unlike Dimcat and humdrumR, only a single piece is handled by a dataframe, not entire corpora.

### Notation

The notation is designed to be easily parsed with existing TOML and TSV parsers.
The implementation uses the python `tomllib` module and pandas' `read_table()` together with regular expressions for parsing time notation.
The raw parsed dataframe is then post-processed into spines by interpreting the time notation and merging dependent columns (such as `vo1` and `id(vo1)` in the example above).
