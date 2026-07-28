# Annotation guidelines — negation probe

Read this fully **before** opening your sample file. Do not discuss items with
the other annotator until both files are filled in.

You have 40 candidate triples. For each one, put `y` or `n` in the `keep`
column. That is the only column you fill. Nothing else.

## The question you are answering

> Is the opposition between `target` and `negation` carried by a negation
> marker, and by nothing else?

Not "is this a contradiction". HebNLI already told us it is a contradiction.
Antonyms, swapped numbers and world knowledge all produce contradictions, and
none of them belong in a negation probe.

A triple is usable when all three hold:

1. **`target` is a proposition.** A full sentence that can be true or false.
2. **`negation` differs from `target` only in polarity.** A negation marker was
   added — `לא`, `אין`, `אינו/אינה/אינם`, `ללא`, `בלתי`, `אי־`, `אף אחד`,
   `מעולם`, `שום` — and nothing else of substance changed.
3. **`paraphrase` preserves the meaning of `target`** in different words, and is
   not a verbatim copy of it.

If a cell is blank or says `WRITE PARAPHRASE`, judge criteria 1 and 2 only —
treat the paraphrase as writable and don't reject on its account.

## Reject — with the patterns that actually occur

These are the failure modes seen in this data. It is machine-translated
MultiNLI, and it breaks in recognisable ways.

**Content dropped.** The negation loses a word that carried part of the claim.
This is the most common failure and the easiest to miss.

> `הוא צריך לשים לב יותר` → `הוא לא צריך לשים לב`

`יותר` is gone, so the two sentences are no longer about the same claim. Watch
for `יותר`, `מאוד`, `רק`, `בעצמי`, `בכל מקום`, `תמיד`, numbers.

Pure discourse markers are *not* content: dropping `ובכן`, `אבל`, `ראשית`,
`כמובן` is fine.

**Not a proposition.** Headings, captions, glossary entries, lists, citations,
sentences truncated mid-clause.

> `תקני דיווח לביקורות ביצועים`

**Type mismatch.** The target is a question and the negation is a statement, or
the reverse. Also exclamation turning into a question.

> `למה הם שלחו אותו?` → `הם לא שלחו אותו`

Negating a question *as a question* is fine and counts as usable:
`מה הממשלה צריכה לעשות?` → `מה הממשלה לא צריכה לעשות?`

**Something other than polarity changed.** An entity, a number, a verb, a tense,
a modal, a person.

> `הוא השתמש במוריס` → `הוא לא השתמש באף אחד`
> `354 הערות` → `שום הערות` is fine; `1,000 דולר` → `1,000,000 דולר` is not
> `יכולים להגיש` → `לא יכלו להגיש` — tense moved, reject

**Scope shifted.** The negation attaches somewhere else than in the target.

> `היה ביקורתי, במיוחד בפולין` → `לא היה ביקורתי בפולין`
> `יש יוצאים מן הכלל מעודדים` → `יש יוצאים מן הכלל שאינם מעודדים`

The second one is subtle: the target says encouraging exceptions exist, the
negation says non-encouraging ones exist. Those are not contradictory at all.

**Negation added to something other than the claim.** Negating the speech act
rather than its content.

> `מה זה? אמר האיש החום` → `האיש החום לא אמר דבר`

**Direction inverted.** The target is already negated, so the "negation" is
actually the positive one.

> `אני לא זוכר איך אומרים...` → `אף פעם לא ידעתי איך אומרים...`

**A clause was added.**

> `זה היה מר אינגלת'ורפ` → `זה היה גבר, אבל זה לא היה מר אינגלת'ורפ`

## Keep, even though it looks imperfect

**Paraphrase and negation share the same rewording.** If both drop the same
clause or use the same synonym, the comparison stays fair — the difference
between them is still only polarity.

**Quantifier negation.** `אף אחד`, `מעולם`, `שום דבר`, `בכל מקום` → `בשום מקום`.
These are negation, just not the bare particle.

**Neg-raising.** `אני חושב שX` → `אני לא חושב שX`. The marker sits on the verb of
thinking rather than on X. Still a clean polarity flip.

**Minor MT noise that doesn't touch the claim** — a stray quotation mark, a
gender slip between paraphrase and negation, a spelling variant of a name.

## When you genuinely cannot decide

Mark `n`. The probe is small on purpose; a borderline item costs more in noise
than it adds in size. But note the id somewhere — if you both marked the same
item as uncertain, that is a signal the guideline needs a line added.
