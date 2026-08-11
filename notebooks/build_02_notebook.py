"""Rebuild notebooks/02_train_nli.ipynb.

Generated rather than hand-edited: two bugs (a broken line continuation and a
mangled em-dash) came from patching the .ipynb JSON by hand, and neither was
visible in the source. This script verifies its own output by decoding it.
"""
import json
from pathlib import Path

BS = chr(92)  # backslash, kept explicit so continuations cannot be mis-escaped
NB = Path("notebooks/02_train_nli.ipynb")


def md(text):
    """A markdown cell from one string; split so each line keeps its newline."""
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def expect(prints, writes):
    """The contract for the cell that follows: what it should print and what it
    leaves behind. Per cell, because "it ran" and "it did the right thing" are
    different questions and only the second one matters here."""
    return md(f"**Prints** &nbsp; {prints}\n\n**Writes** &nbsp; {writes}")


cells = [
    md(
        "# Fine-tune a Hebrew NLI model on clean HebNLI  ·  Person B\n"
        "\n"
        "`nli_rerank.py` defaults to `oriel9p/AlephBERT-FT-HebNLI-LCHAIM`, fine-tuned on all of\n"
        "HebNLI — including the rows the probe was mined from. It has already seen our\n"
        "(target, negation) pairs labelled `contradiction`, so its probe scores are partly\n"
        "recall rather than judgement. This notebook builds the replacement.\n"
        "\n"
        "**Runs in two places:** the Colab web UI, and VS Code with the Colab extension. In both\n"
        "the kernel is a remote Google VM — your local files are *not* there, which is why the\n"
        "first cells clone the repo. Sections 1–2 are CPU-only; the GPU is needed from section 3.\n"
        "\n"
        "The two environments differ in Colab's *frontend* features — stored secrets, Drive\n"
        "mounting, browser downloads. Every cell that uses one falls back rather than failing.\n"
        "\n"
        "Run cell by cell, not Run All. Each cell is preceded by what it should print and what\n"
        "it leaves behind; if the output disagrees, stop there."
    ),

    md(
        "## Where everything ends up\n"
        "\n"
        "The kernel is a Google VM. Three destinations, and only one survives a runtime reset:\n"
        "\n"
        "| what | size | where | survives a reset? |\n"
        "|---|---|---|---|\n"
        "| cloned repo, `data/raw/*.jsonl` | ~400 MB | VM disk | no |\n"
        "| smoke-run checkpoint | ~480 MB | VM disk | no |\n"
        "| full-run model + 2 epoch checkpoints | ~480 MB each | Drive, **if** `--out` points there | yes |\n"
        "| `results/*.json`, `results/nli_train.csv` | a few KB | VM disk, then downloaded | via git |\n"
        "\n"
        "Nothing reaches your own machine until the final cell."
    ),

    md("## 1. Setup"),

    expect("Nothing. This cell only defines a helper.", "Nothing."),

    code(
        "# Secrets three ways: Colab's store, then the environment, then a prompt. The VS Code\n"
        "# extension cannot reach Colab's secret store, so the fallbacks are what make this\n"
        "# notebook portable. getpass also keeps the token out of the saved output.\n"
        "import os, subprocess, getpass\n"
        "\n"
        "def get_secret(name: str) -> str:\n"
        "    try:\n"
        "        from google.colab import userdata\n"
        "        value = userdata.get(name)\n"
        "        if value:\n"
        "            print(f'{name}: from Colab secrets')\n"
        "            return value.strip()\n"
        "    except Exception:\n"
        "        pass\n"
        "    value = os.environ.get(name)\n"
        "    if value:\n"
        "        print(f'{name}: from environment')\n"
        "        return value.strip()\n"
        "    return getpass.getpass(f'{name}: ').strip()"
    ),

    expect(
        "Python and torch versions, the GPU name, and which `google.colab` modules import. "
        "`cuda: True` plus a T4 are the two things to confirm before going further.",
        "Nothing."
    ),

    code(
        "# What are we running on? Answers 'will this work here' before anything slow.\n"
        "import platform\n"
        "print('python      ', platform.python_version())\n"
        "print('cwd         ', os.getcwd())\n"
        "try:\n"
        "    import torch\n"
        "    print('torch       ', torch.__version__, '| cuda:', torch.cuda.is_available())\n"
        "    if torch.cuda.is_available():\n"
        "        print('gpu         ', torch.cuda.get_device_name(0))\n"
        "except ImportError:\n"
        "    print('torch        not installed yet')\n"
        "for mod in ('google.colab.userdata', 'google.colab.drive', 'google.colab.files'):\n"
        "    try:\n"
        "        __import__(mod)\n"
        "        print(f'{mod:24s} available')\n"
        "    except Exception as exc:\n"
        "        print(f'{mod:24s} NOT available ({type(exc).__name__})')"
    ),

    expect(
        "`GH_TOKEN:` and where it came from, pip's log, then the last 3 commits — the top one "
        "should be the newest `nli:` commit.",
        "The repo at `/content/hebrew-negation-embeddings` on the VM. The working directory "
        "moves into it, so every path after this is relative to the repo root."
    ),

    code(
        "OWNER, REPO, BRANCH = 'ItayBoros', 'hebrew-negation-embeddings', 'main'\n"
        "\n"
        "gh_token = get_secret('GH_TOKEN')\n"
        "url = f'https://{gh_token}@github.com/{OWNER}/{REPO}.git'\n"
        "\n"
        "if not os.path.exists(REPO):\n"
        "    subprocess.run(['git','clone','-q','--branch',BRANCH,url,REPO], check=True)\n"
        "os.chdir(REPO if os.path.basename(os.getcwd()) != REPO else '.')\n"
        "subprocess.run(['git','pull','-q','origin',BRANCH], check=True)\n"
        "\n"
        "# drop the token from the stored remote so it is not left on the VM's disk\n"
        "subprocess.run(['git','remote','set-url','origin',\n"
        "                f'https://github.com/{OWNER}/{REPO}.git'], check=True)\n"
        "del gh_token, url\n"
        "\n"
        "!pip install -q -r requirements.txt\n"
        "!git log --oneline -3"
    ),

    expect(
        "`27/27 checks passed`, then `all pipeline checks passed`, `all projection checks "
        "passed`, `all NLI data checks passed`. Anything else: stop here, because everything "
        "downstream inherits the fault.",
        "Nothing."
    ),

    code(
        "# offline checks first - seconds, no network, no GPU\n"
        "!python -m src.data.negation --selftest\n"
        "!python -m tests.test_data_pipeline | tail -3\n"
        "!python -m tests.test_projection | tail -3\n"
        "!python -m tests.test_nli_data | tail -3"
    ),

    md(
        "## 2. Data\n"
        "\n"
        "HebNLI's repo card marks it private, so the download needs a token; `src/data/hebnli.py`\n"
        "reads `HF_TOKEN` from the environment.\n"
        "\n"
        "This has already been run once locally. The cells below should reproduce it exactly —\n"
        "same code, same data, so a different answer means something is wrong:\n"
        "\n"
        "| split | loaded | promptID filter | text audit | kept |\n"
        "|---|---|---|---|---|\n"
        "| train | 300,067 | −2,068 | −9 | 297,990 |\n"
        "| val | 1,999 | −9 | −1 | 1,989 |\n"
        "| test | 884 | −1 | 0 | 883 |"
    ),

    expect("`HF_TOKEN:` and where it came from.", "Nothing. The token stays in memory."),

    code("os.environ['HF_TOKEN'] = get_secret('HF_TOKEN')"),

    expect(
        "Four lines per split. Train: `rows 300067`, `prompts 100390`, "
        "`prompts with e/n/c 91630`. Then `rows 1999` for val, `rows 884` for test.",
        "`data/raw/hebnli_{train,val,test}.jsonl` on the VM, about 400 MB total. Gitignored — "
        "regenerate it, never commit it."
    ),

    code(
        "!python -m src.data.hebnli --split train --out data/raw/hebnli_train.jsonl\n"
        "!python -m src.data.hebnli --split val   --out data/raw/hebnli_val.jsonl\n"
        "!python -m src.data.hebnli --split test  --out data/raw/hebnli_test.jsonl"
    ),

    expect(
        "Per split: `held-out promptIDs 689`, `probe sentences 907`, a three-stage funnel, and "
        "the text-overlap count with up to 5 example rows. Train must end at `297990`, "
        "val `1989`, test `883`.",
        "`data/raw/hebnli_{split}_clean.jsonl` — the training data, on the VM, gitignored.<br>"
        "`results/nli_data_{split}.json` — the manifest, a few KB, **committed**. It records "
        "every filter count and every text-overlap hit, and is what the report cites."
    ),

    code(
        "# Two filters. The promptID list is Itay's 689 held-out prompts; the text audit catches\n"
        "# probe sentences reachable under a *different* promptID, which an id filter cannot see.\n"
        "# All three splits: val shares prompts with train, so an unfiltered val would measure\n"
        "# validation accuracy on rows the probe itself came from.\n"
        "# One line each, no loop - IPython rewrites `!` magics line by line, so a backslash\n"
        "# continuation inside a for-body reaches the Python parser and fails.\n"
        "!python -m src.nli.prepare_data --source data/raw/hebnli_train.jsonl --split train --out data/raw/hebnli_train_clean.jsonl\n"
        "!python -m src.nli.prepare_data --source data/raw/hebnli_val.jsonl   --split val   --out data/raw/hebnli_val_clean.jsonl\n"
        "!python -m src.nli.prepare_data --source data/raw/hebnli_test.jsonl  --split test  --out data/raw/hebnli_test_clean.jsonl"
    ),

    expect("Three lines, one per split, matching the table above.",
           "Nothing. It only reads the manifests back."),

    code(
        "import json\n"
        "for split in ('train', 'val', 'test'):\n"
        "    m = json.load(open(f'results/nli_data_{split}.json', encoding='utf-8'))\n"
        "    f = m['funnel']\n"
        "    print(f\"{split:6s} loaded={f['loaded']:>7}  id_filter=-{f['loaded']-f['prompt_id_clean']:<5}\"\n"
        "          f\"  text_audit=-{m['text_overlap']['rows_dropped']:<3}  kept={m['rows_written']}\")"
    ),

    md(
        "## 3. Smoke run · GPU from here\n"
        "\n"
        "2000 rows, one epoch, a few minutes. It exercises tokenising, the label map, the\n"
        "training loop, checkpoint saving and the manifest before hours are committed to any of\n"
        "them. `train_nli.py` has never run on a GPU or against Colab's transformers version, so\n"
        "this is where a surprise would surface — cheaply.\n"
        "\n"
        "Deliberately on the VM's own disk with no epoch checkpoints: the weights are throwaway."
    ),

    expect(
        "The GPU name, then `pair encoding pair_without_segment_ids` (alephbert-base has no "
        "segment embeddings), a `[warn] smoke run` line, a progress bar, and `val accuracy` / "
        "`val macro F1`. Those two are meaningless at 2000 rows and one epoch, and are flagged "
        "`smoke_run` in the results so they cannot be mistaken for findings.",
        "`checkpoints/alephbert-hebnli-clean/` on the VM, ~480 MB, throwaway.<br>"
        "`results/nli_train_alephbert.json` and one row in `results/nli_train.csv`."
    ),

    code(
        "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
        "\n"
        "!python -m src.nli.train_nli --base alephbert " + BS + "\n"
        "    --train data/raw/hebnli_train_clean.jsonl " + BS + "\n"
        "    --val data/raw/hebnli_val_clean.jsonl " + BS + "\n"
        "    --max-train 2000 --epochs 1"
    ),

    md(
        "## 4. Full run\n"
        "\n"
        "Hours, over ~298k rows. Two things decide whether that survives:\n"
        "\n"
        "**Where the weights go.** `/content` is wiped when the runtime resets, so Drive is the\n"
        "only durable option — and mounting Drive is a Colab-frontend feature that may not work\n"
        "under the VS Code extension. The next cell tries, reports honestly, and picks a path.\n"
        "\n"
        "**`--save-epochs`.** Writes a resumable checkpoint per epoch, keeping the two newest.\n"
        "Without it nothing exists on disk until training completes. If the session dies, re-run\n"
        "the training cell unchanged and it resumes from the newest; `--fresh` starts over.\n"
        "\n"
        "Swap models with `--base alephbertgimmel`. The key names both the checkpoint directory\n"
        "and the manifest, so two runs cannot overwrite each other."
    ),

    expect(
        "Either Drive's mount confirmation, or four `[warn]` lines. Either way the last line "
        "reads `checkpoint dir: ... | durable: True/False`. **Read that line** — it is the "
        "difference between a disconnect costing one epoch and costing the whole run.",
        "Mounts Drive at `/content/drive` if it can, and sets `CKPT`. Nothing else."
    ),

    code(
        "# Drive if we can get it, VM disk if we cannot - but say which, loudly, because the\n"
        "# consequence is not visible until something goes wrong.\n"
        "CKPT = '/content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean'\n"
        "try:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    ON_DRIVE = True\n"
        "except Exception as exc:\n"
        "    ON_DRIVE = False\n"
        "    CKPT = 'checkpoints/alephbert-hebnli-clean'\n"
        "    print(f'[warn] Drive unavailable ({type(exc).__name__}: {exc})')\n"
        "    print('[warn] checkpoints go to the VM disk and die with the runtime.')\n"
        "    print('[warn] epoch checkpoints still protect against a crashed cell, not a reset.')\n"
        "    print('[warn] move the finished model off the VM before the session ends.')\n"
        "print('checkpoint dir:', CKPT, '| durable:', ON_DRIVE)"
    ),

    expect(
        "The same shape as the smoke run without the `[warn] smoke run` line, over ~298k rows "
        "instead of 2000. `resuming from ...` appears at the top if it picked up a checkpoint. "
        "The `val accuracy` here is a real number worth reporting.",
        "`$CKPT/` — the final model, tokenizer and `config.json` carrying the label names.<br>"
        "`$CKPT/_trainer/checkpoint-N/` — two epoch checkpoints, for resuming only.<br>"
        "`results/nli_train_alephbert.json`, and a second row in `nli_train.csv` beside the "
        "smoke row rather than replacing it."
    ),

    code(
        "# re-run this exact cell after a disconnect - it resumes from the newest checkpoint\n"
        "!python -m src.nli.train_nli --base alephbert " + BS + "\n"
        "    --train data/raw/hebnli_train_clean.jsonl " + BS + "\n"
        "    --val data/raw/hebnli_val_clean.jsonl " + BS + "\n"
        "    --save-epochs --out {CKPT}"
    ),

    md(
        "## 5. Verify, then keep the results\n"
        "\n"
        "`check_nli_labels` runs six obvious Hebrew pairs and prints the names from\n"
        "`config.id2label` next to what the model predicts. For the released checkpoint that\n"
        "discovers an undocumented mapping; for ours it confirms the names we wrote survived\n"
        "training and describe what the model actually does — a config can say anything."
    ),

    expect(
        "The model path, `encoding pair`, the label names — expect "
        "`{0: 'entailment', 1: 'neutral', 2: 'contradiction'}` — then six pairs each marked "
        "`[ok]` or `[MISMATCH]`, and a tally. A low tally means the indices are wrong, not the "
        "model. The released checkpoint scores 6/6.",
        "Nothing."
    ),

    code(
        "!python -m src.interventions.check_nli_labels " + BS + "\n"
        "    --model {CKPT} --subfolder \"\""
    ),

    expect(
        "One row per training configuration — the smoke run and the full run side by side, "
        "with `smoke_run` telling them apart.",
        "Nothing."
    ),

    code("import pandas as pd\npd.read_csv('results/nli_train.csv')"),

    expect(
        "Five browser downloads, or the files printed inline when that is unavailable.",
        "**Your machine**, finally — as downloads, or as text in the saved notebook. These five "
        "are what gets committed. The weights are not among them."
    ),

    code(
        "# Browser download is Colab-frontend only. Under VS Code, print the contents instead so\n"
        "# the numbers can be copied straight out of the saved notebook.\n"
        "RESULTS = ['results/nli_train.csv', 'results/nli_train_alephbert.json',\n"
        "           'results/nli_data_train.json', 'results/nli_data_val.json',\n"
        "           'results/nli_data_test.json']\n"
        "try:\n"
        "    from google.colab import files\n"
        "    for path in RESULTS:\n"
        "        files.download(path)\n"
        "except Exception as exc:\n"
        "    print(f'[warn] browser download unavailable ({type(exc).__name__}) - contents below')\n"
        "    for path in RESULTS:\n"
        "        print(f'\\n===== {path} =====')\n"
        "        print(open(path, encoding='utf-8').read())"
    ),

    md(
        "Commit the result files from your machine with the `nli:` prefix.\n"
        "\n"
        "**Still open after this notebook:** `nli_rerank.py` now has a `pair` encoding mode and\n"
        "reads its labels from `config.id2label`, so it *can* load this checkpoint — but `lam`\n"
        "still defaults to 1.0, meaning pure NLI with the embedder's cosine contributing nothing.\n"
        "Tuning λ on the probe's train split is the next piece."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# -- verify by decoding, never by trusting the write ------------------------
loaded = json.loads(NB.read_text(encoding="utf-8"))
problems = []
for i, cell in enumerate(loaded["cells"]):
    source = "".join(cell["source"])
    if "�" in source:
        problems.append(f"cell {i}: replacement char (encoding damage)")
    if cell["cell_type"] != "code":
        continue
    in_block = False
    for line in source.split("\n"):
        if line.rstrip().endswith(":") and not line.startswith((" ", "\t")):
            in_block = True
        # backslash + literal 'n' is the classic mis-escape; it is never intended
        if line.rstrip().endswith(BS + "n"):
            problems.append(f"cell {i}: continuation is backslash-n, not a newline: {line!r}")
        # a magic cannot continue across lines inside a Python block
        if in_block and line.lstrip().startswith("!") and line.rstrip().endswith(BS):
            problems.append(f"cell {i}: `!` continuation inside a block: {line!r}")

n_code = sum(1 for c in loaded["cells"] if c["cell_type"] == "code")
print(f"cells: {len(loaded['cells'])} ({n_code} code)")
print("\n".join(problems) if problems else "no encoding or escaping problems")
