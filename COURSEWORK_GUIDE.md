# CM3604 Deep Learning – Fact Checking Coursework Guide

## What you actually submit
According to the coursework brief, submit:
1. your written PDF (mainly Part 3 error analysis and your method/experiments), and
2. `factcheck.py`.

Do **not** modify `factchecking_main.py` for the submission.

## Required environment
Use Python 3.10 and the exact package versions in `requirements.txt`.

Example setup:

```bash
python3.10 -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

You do not need the spaCy English model for the two required modes. It is only needed for the optional `parsing` mode.

## Project layout

```text
project/
├── factcheck.py
├── factchecking_main.py
├── requirements.txt
├── data/
│   ├── dev_labeled_ChatGPT.jsonl
│   └── passages_bm25_ChatGPT_humfacts.jsonl
├── tune_word_overlap.py          # helper only; do not need to submit
└── error_analysis_helper.py      # helper only; do not need to submit
```

## Part 1 – Word overlap
Run:

```bash
python factchecking_main.py --mode word_overlap
```

Implemented method:
- lowercase + regex tokenization
- remove a fixed list of common English stop words
- Porter stemming
- score = fraction of the fact's content tokens appearing in a retrieved passage
- take the maximum score over the retrieved passages
- classify as Supported when the score is at least ~0.6333

On the supplied data in this workspace, this gives:
- 182 / 221 correct
- accuracy = 0.8235 (82.35%)
- S: precision ≈ 0.8230, recall ≈ 0.8304, F1 ≈ 0.8267
- NS: precision ≈ 0.8241, recall ≈ 0.8165, F1 ≈ 0.8203

Why recall rather than Jaccard/cosine on raw passage words?
Wikipedia passages are much longer than the fact. A symmetric metric can unfairly penalise a good passage for having many extra words. Content-word recall instead asks: "how much of the fact is covered by this evidence?"

To demonstrate threshold tuning:

```bash
python tune_word_overlap.py
```

## Part 2 – Textual entailment
Run on CPU:

```bash
python factchecking_main.py --mode entailment
```

Run on a CUDA-enabled PyTorch install/GPU:

```bash
python factchecking_main.py --mode entailment --cuda
```

Implemented method:
1. Split each retrieved FActScore passage into sentence-like evidence units.
2. Rank sentences using the cheap Part 1 lexical overlap score.
3. Prune very low-overlap candidates to reduce expensive transformer calls.
4. Run DeBERTa NLI on retained premise–hypothesis pairs.
5. Convert logits to probabilities with softmax.
6. Read the entailment class robustly from the model config (fallback to class index 0, matching the coursework's stated class order).
7. Predict Supported if any candidate has P(entailment) >= 0.50.
8. Use early exit, `torch.no_grad()`, explicit object deletion, garbage collection, and CUDA cache cleanup to reduce memory pressure.

The actual DeBERTa weights must be downloaded by Hugging Face the first time you run this mode.

## Part 3 – Error analysis
The brief asks you to inspect up to 10 false positives and 10 false negatives, create 2–4 meaningful recurring error categories, give counts by category, and discuss three examples in detail.

After Part 2 runs, generate candidate errors with:

```bash
python error_analysis_helper.py --cuda
```

or omit `--cuda` on CPU. It writes `entailment_errors.json` containing the first 10 examples of each error type plus the evidence passages.

Possible categories must be chosen from the errors you actually observe. Examples of useful *types of categories to look for* include:
- entity/attribute confusion,
- numerical or date mismatch,
- inference requiring context spread across sentences,
- lexical/semantic ambiguity or negation.

Do not claim a category unless your actual errors support it.

## Final pre-submission checks

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

Both commands must finish without errors. If you use CUDA during development, also test the non-CUDA entailment command at least once if practical, because the coursework's required command is written without `--cuda`.
