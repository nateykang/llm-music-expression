# Music Composition Toolkit

Write a Python script that creates a `music21` Score and binds it to a top-level
variable named `score`. Do not write any files or call `.show()`/`.write()` — just
construct `score`; the surrounding harness exports it to MIDI, audio, and MusicXML.

## Available Libraries

- `music21` — Full music representation: notes, chords, rests, measures, parts, instruments, dynamics, tempo, key/time signatures, plus music theory utilities (scales, intervals, roman numerals, transposition)

### Standard Imports

```python
from music21 import (
    stream, note, chord, clef, key, meter, tempo,
    instrument, dynamics, articulations, expressions,
    tie, duration, metadata, interval, scale, roman,
)
```

## music21 Composition Guide

### Core Hierarchy

A music21 Score contains one or more Parts (one per instrument). Each Part contains Measures (one per bar), and each Measure contains note-level elements: Notes, Chords, and Rests.

### Creating a Score

Create a `stream.Score`, then create a `stream.Part` for each instrument. Insert the instrument, tempo mark, key signature, and time signature into the Part at offset 0. Create `stream.Measure` objects (with a `number` argument), append note-level elements to them, then append the measures to the Part. Finally, append each Part to the Score and bind the Score to the top-level variable `score`.

To set title and composer, assign a `metadata.Metadata` object: `score.metadata = metadata.Metadata(title='...', composer='...')`.

### Duration System

Durations are set via `quarterLength` (number of quarter notes):

| quarterLength | Duration |
|---|---|
| 4.0 | Whole note |
| 3.0 | Dotted half note |
| 2.0 | Half note |
| 1.5 | Dotted quarter note |
| 1.0 | Quarter note |
| 0.75 | Dotted eighth note |
| 0.5 | Eighth note |
| 0.25 | 16th note |
| 0.125 | 32nd note |

For triplets, use `duration.Tuplet(3, 2)` or set quarterLength directly (2/3 for a quarter triplet, 1/3 for an eighth triplet, etc.).

### Notes, Chords, and Rests

Create a Note with `note.Note(pitch, quarterLength=duration)`, where pitch is a string like `'C4'`, `'F#4'`, or `'Bb3'`. Set loudness via `note.volume.velocity` (0--127).

Create a Chord with `chord.Chord(pitches, quarterLength=duration)`, where pitches is a list of pitch strings such as `['C4', 'E4', 'G4']`.

Create a Rest with `note.Rest(quarterLength=duration)`.

Each music21 element (Note, Chord, Rest, etc.) can only belong to one stream. To reuse an element, create a new instance or use `copy.deepcopy()`. Do not use `.clone()` — it does not exist.

### Tempo, Key, and Time Signature

Insert tempo, key, and time signature objects into a Part at offset 0 using the `insert` method. Use `tempo.MetronomeMark(number=bpm)` for tempo, `key.Key(tonic, mode)` for key signature (e.g., tonic `'E'` and mode `'minor'`), and `meter.TimeSignature(ratio)` for time signature (e.g., `'3/4'`).

For mid-piece changes, insert new tempo, key, or time signature objects at the appropriate offset within the Part.

### Instruments

Each Part should have one instrument inserted at offset 0. Available instruments in the `instrument` module and their General MIDI program numbers:

- Piano (0), Harpsichord (6), Organ (19)
- Violin (40), Viola (41), Violoncello (42) (not Cello), Contrabass (43)
- AcousticGuitar (25), ElectricGuitar (27), AcousticBass (32), ElectricBass (33)
- Flute (73), Oboe (68), Clarinet (71), Bassoon (70)
- Horn (60), Trumpet (56), Trombone (57), Tuba (58)
- Timpani (47), Harp (46)
- SopranoSaxophone (64), AltoSaxophone (65), TenorSaxophone (66)
- StringInstrument (48)

For any General MIDI program number not listed above, create a generic `instrument.Instrument()` and set its `midiProgram` attribute to the desired program number, then insert it into the Part.

### Dynamics

Insert dynamic markings into a Measure or Part at a specific offset using `dynamics.Dynamic(marking)`. Available markings: pppp, ppp, pp, p, mp, mf, f, ff, fff, ffff, sf, sfz, fp.

### Ties, Articulations, and Expressions

To tie notes across measures, set the `tie` attribute on the first note to `tie.Tie('start')` and on the continuation note to `tie.Tie('stop')`.

Articulations are appended to a note's `articulations` list. Available articulations include Staccato, Accent, Tenuto, and StrongAccent (all from the `articulations` module).

Note that Fermata is an expression, not an articulation. Append `expressions.Fermata()` to a note's `expressions` list.

### Piano Grand Staff

For piano music, create two Parts and insert a `clef.BassClef()` into the left-hand Part at offset 0. Giving both Parts the same `partName` groups them as a grand staff (treble + bass clef) in the engraved MusicXML.

### Music Theory Helpers

The `scale` module provides scale objects (MajorScale, MinorScale, HarmonicMinorScale, MelodicMinorScale, DorianScale, MixolydianScale, PhrygianScale, WholeToneScale, etc.). Call `getPitches(startPitch, endPitch)` on a scale to retrieve its pitches within a range.

Use `roman.RomanNumeral(figure, key)` to build chords from scale degrees. The resulting object's `pitches` attribute gives the chord tones.

The `interval` module provides `Interval(name)` (e.g., `'P5'` for a perfect fifth) with a `transposeNote` method. Notes also support in-place transposition via `note.transpose(intervalName, inPlace=True)`.
