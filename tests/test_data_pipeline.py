"""
Offline check for the probe pipeline.  ===  PERSON A  ===

Runs the whole mine -> review -> finalize -> validate path on a small
HebNLI-shaped fixture. No network, no models, no `datasets` install needed, so
it is safe to run before every push:

    python -m tests.test_data_pipeline

The fixture is built to exercise the failure modes that matter, not just the
happy path. Each prompt id below is there for a reason:

  1001  particle negation (לא)                     -> keep
  1002  existential (יש -> אין)                     -> keep
  1003  copular (אינה)                              -> keep
  1004  privative (ללא)                             -> keep
  2001  antonym קל/קשה, no negation marker          -> drop  (contradiction != negation)
  2002  number swap 1963/1998                       -> drop
  2003  מלאה/ריקה — `מלא` must not strip to `לא`     -> drop
  2004  one-word sentence                           -> drop  (too short)
  2005  entailment sibling is itself negated        -> keep, paraphrase blanked
  2006  quantifier (אף אחד לא)                      -> keep
  2007  premise already negated (ללא), no new marker -> drop
  2008  negation keeps every word and adds four more -> drop  (containment 1.0
        but not a polarity flip: "מי טעה לגבי קוסובו" ->
        "אני לא מעוניין לדעת מי טעה לגבי קוסובו")
  2009  entailment sibling is a verbatim copy       -> keep, paraphrase blanked

2008 and 2009 come from real HebNLI output, not imagination — both patterns
showed up in the first live mining run over the 300k-row train split.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path

from src.data import hebnli
from src.data.build_probe import (
    STAGES, mine, read_review, to_probe_items, validate_rows, write_review,
)
from src.data.negation import _selftest as negation_selftest
from src.schema import load_probe, save_probe

FIXTURE = Path(__file__).parent / "fixtures" / "hebnli_sample.jsonl"

EXPECTED_KEPT = {"hn1001", "hn1002", "hn1003", "hn1004", "hn2005", "hn2006", "hn2009"}
EXPECTED_DROPPED = {"hn2001", "hn2002", "hn2003", "hn2004", "hn2007", "hn2008"}
NEEDS_PARAPHRASE = {"hn2005", "hn2009"}


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


def main() -> int:
    failures: list = []

    print("== negation module ==")
    if negation_selftest():
        failures.append("negation self-test failed")

    print("\n== load fixture ==")
    rows = hebnli.load(str(FIXTURE))
    check(len(rows) == 22, f"expected 22 rows, got {len(rows)}", failures)
    grouped = hebnli.group_by_prompt(rows)
    check(len(grouped) == 13, f"expected 13 prompts, got {len(grouped)}", failures)
    check(
        grouped["1001"]["contradiction"].premise_he == grouped["1001"]["entailment"].premise_he,
        "siblings of one promptID should share a premise",
        failures,
    )

    print("\n== mine ==")
    candidates, funnel = mine(rows)
    got = {c["id"] for c in candidates}
    check(got == EXPECTED_KEPT, f"kept {sorted(got)}, expected {sorted(EXPECTED_KEPT)}", failures)
    check(
        not (got & {i for i in EXPECTED_DROPPED}),
        "a candidate that should have been filtered out survived",
        failures,
    )
    blank = {c["id"] for c in candidates if not c["paraphrase"]}
    check(blank == NEEDS_PARAPHRASE,
          f"expected {sorted(NEEDS_PARAPHRASE)} to need a paraphrase, got {sorted(blank)}", failures)
    print(funnel.report(STAGES))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        review = tmp / "review.csv"
        write_review(candidates, review)

        # BOM survives the round trip, otherwise Hebrew breaks in Excel
        check(review.read_bytes().startswith(b"\xef\xbb\xbf"), "review.csv lost its BOM", failures)

        print("\n== simulate the human pass ==")
        reviewed = read_review(review)
        hand_written = {
            "hn2005": "הכביש פנוי לנסיעה הבוקר.",
            "hn2009": "היקף שירות הדואר נתון לערעור.",
        }
        for r in reviewed:
            r["keep"] = "y"
            if r["id"] in hand_written:
                r["paraphrase"] = hand_written[r["id"]]
        # rows a careless annotator might produce
        reviewed.append({**reviewed[0], "id": "hnNOMARK",
                         "negation": "הרופא המליץ בחום על הניתוח למטופל."})
        reviewed.append({**reviewed[0], "id": "hnNEGPARA",
                         "paraphrase": "הרופא לא המליץ על הניתוח."})
        reviewed.append({**reviewed[0], "id": "hnEMPTY", "paraphrase": ""})
        reviewed.append({**reviewed[0], "id": "hnSKIP", "keep": "n"})

        good, problems = validate_rows(reviewed)
        good_ids = {r["id"] for r in good}
        check(good_ids == EXPECTED_KEPT, f"validation kept {sorted(good_ids)}", failures)
        check(len(problems) == 3, f"expected 3 rejections, got {len(problems)}: {problems}", failures)

        print("\n== finalize ==")
        items = to_probe_items(good, test_size=0.5, seed=17, source="hebnli")
        check(len(items) == 7, f"expected 7 probe items, got {len(items)}", failures)
        check(
            all(it.split in ("train", "test") for it in items),
            "every item needs a train/test split",
            failures,
        )
        check(
            {it.note for it in items} == {"particle", "existential", "privative", "quantifier"},
            f"unexpected strata: {sorted({it.note for it in items})}",
            failures,
        )

        # the split must be reproducible — the projection is fitted on it
        again = to_probe_items(good, test_size=0.5, seed=17, source="hebnli")
        check(
            [(i.id, i.split) for i in items] == [(i.id, i.split) for i in again],
            "the split is not deterministic across runs",
            failures,
        )
        shifted = to_probe_items(good, test_size=0.5, seed=99, source="hebnli")
        check(
            [(i.id, i.split) for i in items] != [(i.id, i.split) for i in shifted],
            "changing the seed did not change the split",
            failures,
        )

        # no id may appear on both sides
        train = {i.id for i in items if i.split == "train"}
        test = {i.id for i in items if i.split == "test"}
        check(not (train & test), f"leak between splits: {train & test}", failures)

        print("\n== round trip through schema.py ==")
        probe = tmp / "probe.jsonl"
        save_probe(items, probe)
        reloaded = load_probe(probe)
        check(reloaded == items, "probe.jsonl did not survive save/load unchanged", failures)

        print("\n== harness still runs on the result ==")
        from src.harness.run_eval import evaluate
        evaluate(str(probe), ["fake"], ["baseline", "projection"], out_csv=str(tmp / "r.csv"))
        written = list(csv.DictReader((tmp / "r.csv").open(encoding="utf-8")))
        check(len(written) == 2, f"expected 2 result rows, got {len(written)}", failures)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print("all pipeline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
