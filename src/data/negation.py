"""
Hebrew negation detection.  ===  PERSON A  ===

Why this file exists
--------------------
David's note on the proposal: **a contradiction is not automatically a
negation.** HebNLI (a Hebrew translation of MultiNLI) labels a pair
`contradiction` whenever the hypothesis cannot be true given the premise —
which covers antonyms, number swaps, entity swaps, and world knowledge, none of
which are negation. The probe is only valid if the opposition is *carried by a
negation marker*.

So before any hand annotation, we need a mechanical filter that answers:
"does the hypothesis introduce a negation marker that the premise did not have?"
That is what this module does. It is deliberately **high-recall, imperfect
precision** — it narrows tens of thousands of pairs down to a few hundred
candidates that a human then accepts, edits, or rejects.

Three tiers of marker
---------------------
Hebrew negation words are heavily ambiguous with ordinary vocabulary, so the
lexicon is tiered by how safe a bare match is:

``strong``  unambiguous on their own              -> לא, אין, אינו/אינה/..., ללא, בלי, בלתי, מעולם
``phrase``  only negative inside a fixed phrase   -> אף אחד, אף פעם, בשום אופן, כלל לא, אי אפשר
``weak``    frequently non-negative, off by default -> אל (imperative), טרם, שום, לאו

`אף` alone means "nose" or "also"; `שום` alone means "garlic"; `כלל` alone means
"rule"; `אל` alone is the preposition "to". Matching them bare would flood the
candidate pool with noise, so they live in ``phrase``/``weak``.

Prefix clitics
--------------
Hebrew glues ו/ש/ה/כ/ל/ב/מ onto the next word, so `לא` also shows up as
`ולא`, `שלא`, `הלא`, `בלא`. We strip up to two leading clitics and re-check the
lexicon. That creates its own trap — `מלא` ("full") and `כלא` ("prison") strip
down to `לא`, and `מאין` usually means "whence", not "there is no". Those sit in
``NEVER_STRIP`` and are never treated as negation.

Self-test:  python -m src.data.negation --selftest
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------

#: Unambiguous negation words. Safe to match as a bare token.
STRONG: Set[str] = {
    "לא",
    "אין",
    # copular negation, full person paradigm
    "איני", "אינני", "אינך", "אינו", "אינה",
    "איננו", "איננה", "אינם", "אינן", "אינכם", "אינכן",
    # privative prepositions
    "ללא", "בלי", "בלעדי", "בלתי",
    # temporal
    "מעולם",
}

#: Only negative inside a fixed collocation. Stored as (first, second) token
#: bigrams; the tokens are matched after clitic stripping.
PHRASE: Set[Tuple[str, str]] = {
    ("אף", "אחד"), ("אף", "אחת"), ("אף", "פעם"), ("אף", "לא"), ("אף", "אינו"),
    ("שום", "דבר"), ("שום", "אדם"), ("שום", "סיבה"), ("שום", "אופן"),
    ("בשום", "אופן"), ("בשום", "פנים"), ("בשום", "מקרה"), ("בשום", "צורה"),
    ("כלל", "לא"), ("בכלל", "לא"), ("כלל", "אין"),
    ("אי", "אפשר"),
    ("מעולם", "לא"), ("מעודו", "לא"),
    ("לאו", "דווקא"),
}

#: Frequently non-negative in ordinary text. Excluded unless include_weak=True.
WEAK: Set[str] = {
    "טרם",     # also "before"
    "שום",     # also "garlic"
    "לאו",
    "מעודו", "מעודה", "מעודי",
}

#: Clitics that can be glued to the front of a word.
PREFIXES: Tuple[str, ...] = ("ו", "ש", "ה", "כ", "ל", "ב", "מ")

#: Words that strip down to a negation word but are not negation.
#: מלא = full, כלא = prison, הלאה = onward, לאט = slowly, לאום = nation,
#: מאין = whence, כלאחר = as if in passing.
NEVER_STRIP: Set[str] = {
    "מלא", "מלאה", "מלאים", "מלאות", "מלאי", "מלאך", "מלאכה",
    "כלא", "כלאי", "כלאיים",
    "הלאה", "לאט", "לאום", "לאומי", "לאומית", "לאומיים",
    "מאין", "כלאחר",
    "אלא", "בלבד", "הלוואי",
}

#: `אי` used as a privative prefix: אי-אפשר, אי-הסכמה, אי-שוויון.
#: Requires a hyphen or maqaf, otherwise `אי` is the noun "island".
AI_PRIVATIVE = re.compile(r"\bאי[־\-]\s*\S")

# Vowel points, cantillation and Hebrew punctuation — but NOT U+05BE MAQAF,
# which is a real hyphen and has to survive for `אי־אפשר` to be recognisable.
NIQQUD = re.compile("[֑-ֽֿׁׂׄ-ׇ]")
NON_WORD = re.compile(r"[^א-תװ-״a-zA-Z0-9־\-]+")


@dataclass(frozen=True)
class Marker:
    """One negation marker found in a sentence."""
    surface: str       # the token as it appeared, e.g. "שאינו"
    lemma: str         # the lexicon entry it reduced to, e.g. "אינו"
    tier: str          # "strong" | "phrase" | "weak"
    index: int         # token position


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

def strip_niqqud(text: str) -> str:
    """Drop vowel points and cantillation; HebNLI is mostly unvocalised but
    stray niqqud would otherwise break exact lexicon matching."""
    return NIQQUD.sub("", unicodedata.normalize("NFC", text))


def tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenisation, keeping the maqaf (־) so that
    `אי־אפשר` survives as a unit for AI_PRIVATIVE."""
    text = strip_niqqud(text)
    text = NON_WORD.sub(" ", text)
    return [t for t in text.split() if t]


def strip_clitics(token: str, max_depth: int = 2) -> List[str]:
    """Return the token plus every form reachable by removing leading clitics.

    `ושלא` -> ["ושלא", "שלא", "לא"]. Members of NEVER_STRIP are returned as-is,
    because stripping them produces a false negation (מלא -> לא).
    """
    forms = [token]
    if token in NEVER_STRIP:
        return forms
    current = token
    for _ in range(max_depth):
        if len(current) <= 2 or current[0] not in PREFIXES:
            break
        current = current[1:]
        if current in NEVER_STRIP:
            break
        forms.append(current)
    return forms


def _reduce(token: str) -> Tuple[str, ...]:
    return tuple(strip_clitics(token))


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def find_markers(text: str, include_weak: bool = False) -> List[Marker]:
    """All negation markers in `text`, in order of appearance.

    A token is reported at most once, at the strongest tier that matches, so a
    bigram hit like (אף, פעם) does not also produce two weak hits.
    """
    tokens = tokenize(text)
    reduced = [_reduce(t) for t in tokens]
    found: List[Marker] = []
    claimed: Set[int] = set()

    # 1. phrase bigrams first — they claim their tokens
    for i in range(len(tokens) - 1):
        for a in reduced[i]:
            for b in reduced[i + 1]:
                if (a, b) in PHRASE:
                    found.append(Marker(f"{tokens[i]} {tokens[i+1]}", f"{a} {b}", "phrase", i))
                    claimed.update({i, i + 1})
                    break
            else:
                continue
            break

    # 2. strong single tokens
    for i, forms in enumerate(reduced):
        if i in claimed:
            continue
        for form in forms:
            if form in STRONG:
                found.append(Marker(tokens[i], form, "strong", i))
                claimed.add(i)
                break

    # 3. weak single tokens, opt-in
    if include_weak:
        for i, forms in enumerate(reduced):
            if i in claimed:
                continue
            for form in forms:
                if form in WEAK:
                    found.append(Marker(tokens[i], form, "weak", i))
                    claimed.add(i)
                    break
        # negative imperative: אל followed by a 2nd-person verb (תלך, תיגע)
        for i, tok in enumerate(tokens[:-1]):
            if i in claimed:
                continue
            if tok == "אל" and tokens[i + 1].startswith("ת"):
                found.append(Marker(f"{tok} {tokens[i+1]}", "אל+imperative", "weak", i))
                claimed.update({i, i + 1})

    # 4. אי- privative
    for m in AI_PRIVATIVE.finditer(strip_niqqud(text)):
        found.append(Marker(m.group(0).strip(), "אי-", "strong", -1))

    return sorted(found, key=lambda mk: (mk.index, mk.tier))


def has_negation(text: str, include_weak: bool = False) -> bool:
    return bool(find_markers(text, include_weak=include_weak))


def added_negation(premise: str, hypothesis: str, include_weak: bool = False) -> List[Marker]:
    """Markers the hypothesis introduces that the premise does not already have.

    This is the filter that matters. A premise that is *already* negated and a
    hypothesis that drops the negation is also interesting, but it inverts the
    triple's direction, so `build_probe` handles that case separately.
    """
    premise_lemmas = {m.lemma for m in find_markers(premise, include_weak=include_weak)}
    return [
        m for m in find_markers(hypothesis, include_weak=include_weak)
        if m.lemma not in premise_lemmas
    ]


# --------------------------------------------------------------------------
# minimal-edit measurement
# --------------------------------------------------------------------------

def _content_tokens(text: str, include_weak: bool = False) -> List[str]:
    """Tokens with negation markers removed, so overlap measures *the rest of
    the sentence* — which is what CONDAQA-style minimal editing constrains."""
    tokens = tokenize(text)
    drop: Set[int] = set()
    for m in find_markers(text, include_weak=include_weak):
        if m.index < 0:
            continue
        drop.add(m.index)
        if m.tier == "phrase" or " " in m.surface:
            drop.add(m.index + 1)  # bigram markers are indexed at their first token
    return [t for i, t in enumerate(tokens) if i not in drop]


def overlap(premise: str, hypothesis: str) -> dict:
    """How close the two sentences are once negation is stripped out.

    ``containment`` — share of premise content tokens that survive in the
    hypothesis. High containment plus an added negation marker is the signature
    of a genuine minimal negation edit; low containment means the hypothesis was
    rewritten and the opposition may come from somewhere else entirely.

    ``added`` — content words the hypothesis introduces. Containment alone does
    not catch a hypothesis that keeps every original word and then piles more on
    top ("מי טעה לגבי קוסובו" -> "אני לא מעוניין לדעת מי טעה לגבי קוסובו"):
    containment is a perfect 1.0, but the opposition is no longer a polarity
    flip. Capping ``added`` is what rejects those.
    """
    a = _content_tokens(premise)
    b = _content_tokens(hypothesis)
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return {"containment": 0.0, "jaccard": 0.0, "len_ratio": 0.0, "added": len(sb)}
    inter = len(sa & sb)
    return {
        "containment": inter / len(sa),
        "jaccard": inter / len(sa | sb),
        "len_ratio": min(len(a), len(b)) / max(len(a), len(b)),
        "added": len(sb - sa),
    }


def same_sentence(a: str, b: str) -> bool:
    """True if the two differ only by punctuation, whitespace or niqqud.

    HebNLI's entailment hypothesis is quite often a verbatim copy of the
    premise. That is a fine entailment and a useless paraphrase — a probe item
    whose target and paraphrase are the same string measures nothing.
    """
    return tokenize(a) == tokenize(b)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

_POSITIVE = [
    "החתול לא ישן על הספה.",
    "אין חלב במקרר.",
    "התרופה אינה מטפלת במחלה.",
    "הוא אמר שלא יגיע.",
    "היא הלכה ולא חזרה.",
    "הפתרון בלתי אפשרי.",
    "הוא נסע ללא רישיון.",
    "מעולם לא ביקרתי שם.",
    "אף אחד לא ענה.",
    "אף פעם לא ראיתי כזה דבר.",
    "בשום אופן לא נסכים.",
    "אי־אפשר להמשיך ככה.",
    "הם אינם מעוניינים בהצעה.",
]

_NEGATIVE = [
    "הכוס מלאה במים.",                 # מלא -> לא
    "הוא ישב בבית הכלא.",              # כלא -> לא
    "נסע לאט בכביש הרטוב.",            # לאט
    "הוא הצביע בעד החוק הלאומי.",      # לאומי
    "האף שלו אדום.",                   # אף = nose
    "הוסיפו שום וגזר למרק.",           # שום = garlic
    "זה כלל ידוע בפיזיקה.",            # כלל = rule
    "הלכתי אל הים.",                   # אל = to
    "אלא שהוא איחר.",                  # אלא
    "הוא רץ הלאה בלי לעצור.",          # הלאה — but בלי IS negation
]


def _selftest() -> int:
    failures = 0
    for s in _POSITIVE:
        if not has_negation(s):
            print(f"[FAIL] expected negation, found none: {s}")
            failures += 1
    for s in _NEGATIVE[:-1]:  # last one legitimately contains בלי
        if has_negation(s):
            got = [f"{m.surface}->{m.lemma}" for m in find_markers(s)]
            print(f"[FAIL] false positive {got}: {s}")
            failures += 1
    if not has_negation(_NEGATIVE[-1]):
        print(f"[FAIL] expected בלי to be caught: {_NEGATIVE[-1]}")
        failures += 1

    # added_negation must ignore negation the premise already had
    prem, hyp = "הוא לא הגיע לפגישה.", "הוא לא הגיע לעבודה."
    if added_negation(prem, hyp):
        print("[FAIL] added_negation fired on a shared marker")
        failures += 1
    prem, hyp = "הוא הגיע לפגישה.", "הוא לא הגיע לפגישה."
    if not added_negation(prem, hyp):
        print("[FAIL] added_negation missed a newly introduced marker")
        failures += 1

    # overlap: minimal edit should score near 1, rewrite should score low
    hi = overlap("הרופא המליץ על הניתוח.", "הרופא לא המליץ על הניתוח.")["containment"]
    lo = overlap("הרופא המליץ על הניתוח.", "אין קשר בין תזונה לספורט.")["containment"]
    if hi < 0.9:
        print(f"[FAIL] minimal edit containment too low: {hi:.2f}")
        failures += 1
    if lo > 0.3:
        print(f"[FAIL] unrelated pair containment too high: {lo:.2f}")
        failures += 1

    total = len(_POSITIVE) + len(_NEGATIVE) + 4
    print(f"\n{total - failures}/{total} checks passed")
    return failures


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(line, "->", [f"{m.surface}({m.tier})" for m in find_markers(line)])
