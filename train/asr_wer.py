"""Shared ASR + WER helpers for Track 2 (Noise Event Removal).

The leaderboard scores Combined = SI-SDR(synthetic) + 100 * dWER(all clips),
where dWER = WER_noisy - WER_enhanced. WER_noisy is a fixed organiser baseline,
so improving the score means lowering WER on OUR enhanced audio -- and a system
that over-suppresses raises WER and scores NEGATIVE. Being able to measure that
locally, before submitting, is the whole game (cf. Track 1, where six
consecutive leaderboard regressions came from validating on the wrong subset).

Reference transcripts carry noise annotations in two styles that must be
stripped before scoring: <bird sound> ... </bird sound> and [breathing].
The organisers' exact normaliser is not published, so treat absolute WER as
indicative and rely on DIFFERENCES between conditions, which is what dWER is.
"""
from __future__ import annotations
import re
import unicodedata

_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")


def normalise(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _TAG.sub(" ", t)          # <bird sound>, </bird sound>, [breathing]
    # Strip punctuation/symbols BY UNICODE CATEGORY. A [^\w\s] class silently
    # deletes Indic combining marks (category Mn) -- Python's \w excludes them --
    # which mangles every Devanagari/Bengali word and inflates WER.
    t = "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def wer_counts(ref: str, hyp: str):
    """-> (edit_distance, n_ref_words) so counts can be micro-averaged."""
    r = normalise(ref).split()
    h = normalise(hyp).split()
    if not r:
        return (0, 0) if not h else (len(h), 0)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)], len(r)


def corpus_wer(pairs):
    """pairs of (ref, hyp) -> micro-averaged WER (total edits / total ref words)."""
    e = n = 0
    for ref, hyp in pairs:
        a, b = wer_counts(ref, hyp)
        e += a
        n += b
    return e / n if n else 0.0
