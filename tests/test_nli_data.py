"""
Offline check for the NLI training-data filters.  ===  PERSON B  ===

Runs both contamination filters over the same small HebNLI-shaped fixture the
probe pipeline uses. No network, no models, no GPU, so it is safe before every
push:

    python -m tests.test_nli_data

What is actually being tested, and why it is worth a test at all: the filters in
`src/nli/prepare_data.py` are the only thing standing between us and a reported
number that is really memorisation (see that module's docstring). A filter that
silently stops filtering looks exactly like a filter that works — the training
run still succeeds, the accuracy still looks fine — so the failure has to be
caught here rather than noticed later.

Four properties, each with a planted failure:

  1. a held-out promptID takes ALL its siblings with it, not just the
     contradiction row (promptID 1001 -> 3 rows, e/n/c)
  2. the post-filter guard raises if a held-out prompt survives — checked by
     disabling the filter, since it cannot happen while the code is correct
  3. the text audit catches a probe sentence that appears under a promptID the
     held-out list does NOT contain (promptID 1003), which is the case an
     identifier-level filter is blind to
  4. the results table keeps one row per configuration, so training a second
     base model cannot silently overwrite the first one's scores

The last one is not a filter, but it fails the same way the others do — quietly,
leaving a file that still looks right.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from src.data import hebnli
from src.nli.prepare_data import (
    find_text_overlap, normalise, prepare, probe_sentences,
)
from src.nli.train_nli import append_result
from src.schema import ProbeItem, save_probe

FIXTURE = Path(__file__).parent / "fixtures" / "hebnli_sample.jsonl"

#: Held out by id. The fixture gives it three siblings, so it also proves the
#: exclusion is by promptID and not by pairID.
HELDOUT = {"1001"}

#: Present in the fixture under promptID 1003, which is NOT held out. This is
#: the leak the id filter cannot see — the same Hebrew string reachable from a
#: different prompt, which happens because HebNLI is machine-translated.
LEAKED_TARGET = "התרופה מטפלת במחלה ביעילות רבה."

#: Same sentence with sloppy internal spacing, so the audit is also exercising
#: `normalise`. A whitespace difference is not a different sentence.
LEAKED_TARGET_SPACED = "התרופה  מטפלת   במחלה ביעילות רבה."

EXPECTED_LOADED = 22
EXPECTED_AFTER_ID = 19          # 22 - 3 siblings of promptID 1001
EXPECTED_AFTER_TEXT = 17        # 19 - 2 rows of promptID 1003


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


def write_probe(path: Path) -> None:
    """A two-item probe: one item whose target leaks into the fixture, one that
    is entirely absent so we can tell filtering from over-filtering."""
    save_probe([
        ProbeItem(
            id="t001", target=LEAKED_TARGET_SPACED,
            paraphrase="התרופה יעילה מאוד נגד המחלה.",
            negation="התרופה אינה מטפלת במחלה ביעילות רבה.",
            source="hebnli", split="test",
        ),
        ProbeItem(
            id="t002", target="משפט שאינו מופיע באוסף כלל.",
            paraphrase="משפט אחר שאינו מופיע באוסף.",
            negation="משפט שלישי שאינו מופיע באוסף.",
            source="handwritten", split="train",
        ),
    ], path)


def main() -> int:
    failures: list = []

    print("== normalise ==")
    check(normalise(LEAKED_TARGET_SPACED) == LEAKED_TARGET,
          "internal whitespace should collapse to a single space", failures)
    check(normalise("  a b  ") == "a b", "outer whitespace should strip", failures)

    print("\n== load fixture ==")
    rows = hebnli.load(str(FIXTURE))
    check(len(rows) == EXPECTED_LOADED,
          f"expected {EXPECTED_LOADED} rows, got {len(rows)}", failures)

    with tempfile.TemporaryDirectory() as tmpdir:
        probe_path = Path(tmpdir) / "probe.jsonl"
        write_probe(probe_path)

        print("\n== probe sentences ==")
        sentences = probe_sentences(probe_path)
        check(len(sentences) == 6, f"expected 6 sentences, got {len(sentences)}", failures)
        check(LEAKED_TARGET in sentences,
              "the spaced target should normalise into the sentence set", failures)

        print("\n== promptID exclusion takes all siblings ==")
        kept, hits, funnel = prepare(rows, HELDOUT, sentences)
        check(funnel.counts["prompt_id_clean"] == EXPECTED_AFTER_ID,
              f"expected {EXPECTED_AFTER_ID} after the id filter, "
              f"got {funnel.counts['prompt_id_clean']}", failures)
        check(not any(r.prompt_id in HELDOUT for r in kept),
              "a held-out promptID survived", failures)

        print("\n== text audit catches a non-held-out leak ==")
        check(len(hits) == 2, f"expected 2 text-overlap hits, got {len(hits)}", failures)
        check(all(h["prompt_id"] == "1003" for h in hits),
              f"hits should all be promptID 1003, got "
              f"{sorted({h['prompt_id'] for h in hits})}", failures)
        check(len(kept) == EXPECTED_AFTER_TEXT,
              f"expected {EXPECTED_AFTER_TEXT} clean rows, got {len(kept)}", failures)

        print("\n== the invariant that matters ==")
        # Whatever the counts, no probe sentence may reach the training file.
        leaked = [r.pair_id for r in kept
                  if normalise(r.premise_he) in sentences
                  or normalise(r.hypothesis_he) in sentences]
        check(not leaked, f"probe text reached the clean set: {leaked[:3]}", failures)

        print("\n== no over-filtering ==")
        # t002's sentences appear nowhere, so nothing may be dropped on its account.
        untouched, absent_hits = find_text_overlap(
            rows, {normalise("משפט שאינו מופיע באוסף כלל.")})
        check(not absent_hits, f"absent sentence matched {len(absent_hits)} rows", failures)
        check(len(untouched) == EXPECTED_LOADED, "no row should have been dropped", failures)

        print("\n== the survivor guard fires ==")
        # It cannot trigger while drop_prompts works, so disable the filter and
        # confirm the guard is what stops a contaminated set, not luck.
        original = hebnli.drop_prompts
        hebnli.drop_prompts = lambda rows, prompt_ids: list(rows)
        try:
            prepare(rows, HELDOUT, sentences)
            check(False, "guard did not raise when a held-out prompt survived", failures)
        except ValueError:
            pass
        finally:
            hebnli.drop_prompts = original

        print("\n== results table keeps one row per configuration ==")
        # The failure this guards against is silent: a second run overwriting
        # the first model's scores, leaving a plausible-looking file with a
        # result missing from it.
        table = Path(tmpdir) / "nli_train.csv"
        base_row = {"base": "alephbert", "smoke_run": False, "lr": 2e-5,
                    "batch_size": 32, "epochs": 2.0, "max_length": 128,
                    "seed": 17, "max_train": "", "accuracy": 0.81}
        check(append_result(dict(base_row), table) == 1, "first run should write 1 row", failures)
        check(append_result(dict(base_row), table) == 1,
              "an identical config should replace, not duplicate", failures)

        gimmel = dict(base_row, base="alephbertgimmel", accuracy=0.84)
        check(append_result(gimmel, table) == 2, "a different base should append", failures)

        smoke = dict(base_row, smoke_run=True, max_train=2000, accuracy=0.55)
        check(append_result(smoke, table) == 3,
              "a smoke run must not overwrite the real run of the same base", failures)

        with table.open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        kept = {r["base"]: r["accuracy"] for r in written if r["smoke_run"] == "False"}
        check(kept.get("alephbert") == "0.81",
              f"alephbert's score was lost: {kept}", failures)
        check(kept.get("alephbertgimmel") == "0.84",
              f"gimmel's score was lost: {kept}", failures)

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall NLI data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
