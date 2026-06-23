"""Measure inductive biases in generated music via standard symbolic metrics.

Every piece (code-gen MusicXML or ABC text) is reduced to an analyzable form, then
a panel of *standard* symbolic-music metrics is computed. The metrics are chosen to
surface what models reach for **by default** — aggregate them over a batch and you
get statements like "model X defaults to minor keys / sparse textures / slow tempi."

Metric provenance (so choices are citable):
  - MusPy (Dong et al., 2020): pitch_class_entropy, scale_consistency,
    pitch_in_scale_rate, polyphony, empty_beat_rate, groove_consistency,
    n_pitches_used, pitch_range.
  - music21: key/mode (Krumhansl-Schmuckler), tempo, note content.
  - valence/arousal: a transparent feature-based affect proxy following the
    Russell circumplex convention in Music Emotion Recognition (mode -> valence;
    tempo + note density -> arousal). A heuristic, not a trained MER model.
"""

from __future__ import annotations

import csv
import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Columns emitted per piece, in order.
FIELDS = [
    "model", "prompt", "mode", "title",
    "key_tonic", "key_mode", "key_confidence",
    "key_declared_tonic", "key_declared_mode", "key_mode_best", "mode_match",
    "scale_consistency", "pitch_class_entropy", "pitch_in_scale_rate",
    "consonance_rate", "chord_tone_rate",
    "polyphony", "n_voices", "empty_beat_rate", "groove_consistency",
    "n_pitches_used", "pitch_range",
    "tempo_bpm", "n_notes", "length_seconds", "note_density",
    "valence", "arousal", "affect_quadrant",
]


def _load(piece: dict, batch_dir: Path, work_dir: Path):
    """Return (muspy.Music, music21.Score) for a piece, or (None, None).

    Everything is routed through MIDI for MusPy (its MIDI reader is robust; its
    MusicXML reader is not). music21 reads the richer source (MusicXML for code-gen)
    for key/tempo.
    """
    import muspy
    from music21 import converter

    midi = work_dir / "piece.mid"
    if piece.get("score"):  # code-gen: MusicXML is the source of truth
        score = converter.parse(str(batch_dir / piece["score"]))
        score.write("midi", fp=str(midi))
    elif piece.get("abc"):  # ABC: render to MIDI with abc2midi (the canonical tool)
        from .render import abc_to_midi

        produced = abc_to_midi(piece["abc"], work_dir)
        if not produced:
            return None, None
        midi = produced
        score = converter.parse(str(midi))
    else:
        return None, None
    try:
        return muspy.read_midi(str(midi)), score
    except Exception:
        return None, None


def _parse_declared_key(abc: str):
    """Parse the model's DECLARED key from the ABC K: header → (tonic, 'major'/'minor').

    This is the model's stated intent (e.g. K:Dmin), unambiguous and free of the
    detection noise that Krumhansl–Schmuckler introduces. Modes are classified by
    their third: Ionian/Lydian/Mixolydian → major; Aeolian/Dorian/Phrygian/Locrian
    and bare 'm'/'min' → minor. Returns (None, None) if there's no usable K: field.
    """
    import re

    if not abc:
        return None, None
    m = re.search(r"(?mi)^[ \t]*K:[ \t]*([A-Ga-g])([#b]?)[ \t]*([A-Za-z]*)", abc)
    if not m:
        return None, None
    tonic = m.group(1).upper() + {"#": "#", "b": "-"}.get(m.group(2), "")
    s = m.group(3).lower()
    if not s or s.startswith(("maj", "ion", "lyd", "mix")):
        mode = "major"
    elif s.startswith(("min", "aeo", "dor", "phr", "loc")) or s[0] == "m":
        mode = "minor"
    else:
        mode = "major"
    return tonic, mode


def _harmony_metrics(score):
    """MuSpike-style harmony metrics from a chordified score → (consonance, chord_tone).

    consonance:  fraction of vertical sonorities (≥2 notes) that are consonant
                 (music21 Chord.isConsonant) — a Pitch-Consonance-Score analog; how
                 harmonically clean the vertical writing is.
    chord_tone:  per measure, the prevailing harmony = the 3 most duration-present
                 pitch classes; the fraction of sounding pitches in that set — a
                 Chord-Tone / Non-Chord-Tone-Ratio analog.
    """
    from music21 import chord as m21chord

    try:
        chords = score.chordify()
        sonorities = list(chords.recurse().getElementsByClass(m21chord.Chord))
    except Exception:
        return None, None
    if not sonorities:
        return None, None

    multi = [c for c in sonorities if len(c.pitches) >= 2]
    consonance = (sum(c.isConsonant() for c in multi) / len(multi)) if multi else None

    hits = total = 0
    measures = list(chords.recurse().getElementsByClass("Measure")) or [chords]
    for m in measures:
        pc_dur: dict[int, float] = {}
        cs = list(m.recurse().getElementsByClass(m21chord.Chord))
        for c in cs:
            for p in c.pitches:
                pc_dur[p.pitchClass] = pc_dur.get(p.pitchClass, 0.0) + float(c.quarterLength or 0)
        if not pc_dur:
            continue
        prevailing = set(sorted(pc_dur, key=lambda k: -pc_dur[k])[:3])
        for c in cs:
            for p in c.pitches:
                total += 1
                hits += p.pitchClass in prevailing
    chord_tone = (hits / total) if total else None
    return (round(consonance, 4) if consonance is not None else None,
            round(chord_tone, 4) if chord_tone is not None else None)


def extract_features(piece: dict, batch_dir: Path) -> dict | None:
    import muspy
    from music21 import tempo as m21tempo

    with tempfile.TemporaryDirectory(prefix="llm_music_an_") as td:
        mus, score = _load(piece, batch_dir, Path(td))
    if mus is None or score is None:
        return None

    def safe(fn, *args):
        try:
            v = float(fn(mus, *args))
        except Exception:
            return None
        return None if v != v else round(v, 4)  # NaN (e.g. 0/0 on a note-less piece) -> None

    consonance_rate, chord_tone_rate = _harmony_metrics(score)

    # Degenerate pieces (empty / all-rest, e.g. a hollow generation) can't be
    # key-analyzed — record them with unknown tonality rather than dropping them.
    try:
        key = score.analyze("key")
        tonic, mode = key.tonic.name, key.mode
        key_conf = round(float(key.tonalCertainty()), 3)
    except Exception:
        tonic, mode, key_conf = "?", "?", None
    mm = score.recurse().getElementsByClass(m21tempo.MetronomeMark).first()
    bpm = float(mm.number) if mm and mm.number else 120.0
    n_notes = len(list(score.recurse().notes))
    length_q = float(score.highestTime) or 1.0          # duration in quarter-notes
    length_s = length_q * 60.0 / bpm
    notes_per_beat = n_notes / length_q                 # tempo-invariant rhythmic density

    resolution = (mus.resolution or 480) * 4  # assume 4 beats/measure for grooving

    # Declared (K: field) vs detected (Krumhansl–Schmuckler) key. Declared is the
    # model's stated intent — more reliable for "what key did it choose" — so it
    # drives the headline metrics; the gap between them is intent-vs-execution.
    decl_tonic, decl_mode = _parse_declared_key(piece.get("abc", ""))
    best_mode = decl_mode or (mode if mode != "?" else None)
    mode_match = None if (not decl_mode or mode == "?") else int(decl_mode == mode)

    # affect proxy: mode -> valence; tempo + rhythmic density -> arousal (Russell
    # circumplex). Density is notes-PER-BEAT (not per-second) so it stays independent
    # of tempo — otherwise the two arousal terms would both encode speed.
    valence = 0 if best_mode in (None, "?") else (1 if best_mode == "major" else -1)
    tempo_norm = max(0.0, min(1.0, (bpm - 50) / (160 - 50)))
    dens_norm = max(0.0, min(1.0, notes_per_beat / 4.0))
    arousal = round(0.6 * tempo_norm + 0.4 * dens_norm, 3)
    quadrant = ("unknown" if valence == 0
                else "happy/excited" if (valence > 0 and arousal >= 0.5)
                else "serene/content" if (valence > 0)
                else "angry/tense" if (arousal >= 0.5)
                else "sad/depressed")

    return {
        "model": piece["model"], "prompt": piece["prompt"],
        "mode": piece.get("mode"), "title": piece.get("title", ""),
        "key_tonic": tonic, "key_mode": mode,
        "key_confidence": key_conf,
        "key_declared_tonic": decl_tonic or "", "key_declared_mode": decl_mode or "",
        "key_mode_best": best_mode or "", "mode_match": "" if mode_match is None else mode_match,
        "scale_consistency": safe(muspy.scale_consistency),
        "pitch_class_entropy": safe(muspy.pitch_class_entropy),
        "pitch_in_scale_rate": safe(muspy.pitch_in_scale_rate),
        "consonance_rate": consonance_rate, "chord_tone_rate": chord_tone_rate,
        "polyphony": safe(muspy.polyphony),
        "n_voices": len(mus.tracks),
        "empty_beat_rate": safe(muspy.empty_beat_rate),
        "groove_consistency": safe(muspy.groove_consistency, resolution),
        "n_pitches_used": safe(muspy.n_pitches_used),
        "pitch_range": safe(muspy.pitch_range),
        "tempo_bpm": round(bpm, 1), "n_notes": n_notes,
        "length_seconds": round(length_s, 1), "note_density": round(notes_per_beat, 2),
        "valence": valence, "arousal": arousal, "affect_quadrant": quadrant,
    }


def analyze_batch(batch_dir: Path) -> list[dict]:
    """Extract features for every successful piece in a batch."""
    manifest = json.loads((batch_dir / "data.json").read_text(encoding="utf-8"))
    rows = []
    for p in manifest["pieces"]:
        if not p.get("ok"):
            continue
        feats = extract_features(p, batch_dir)
        if feats:
            rows.append(feats)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
