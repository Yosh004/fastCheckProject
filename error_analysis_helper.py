"""Generate false-positive/false-negative examples for Part 3.
Not required for submission. Run after the entailment implementation works.
"""
import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentFactChecker, EntailmentModel

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def read_data():
    fact_to_passages = {}
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            fact_to_passages[item["sent"]] = item["passages"]

    examples = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for sent in item.get("annotations") or []:
                for fact in sent.get("human-atomic-facts") or []:
                    text = fact["text"]
                    if text in fact_to_passages:
                        gold = "NS" if fact["label"] == "IR" else fact["label"]
                        examples.append((text, gold, fact_to_passages[text]))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", default="entailment_errors.json")
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    print("CUDA:", use_cuda)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    if use_cuda:
        model.to("cuda")

    checker = EntailmentFactChecker(EntailmentModel(model, tokenizer, use_cuda))
    false_positives = []
    false_negatives = []

    for i, (fact, gold, passages) in enumerate(read_data(), start=1):
        pred = checker.predict(fact, passages)
        if gold == "NS" and pred == "S":
            false_positives.append({"fact": fact, "gold": gold, "pred": pred, "passages": passages})
        elif gold == "S" and pred == "NS":
            false_negatives.append({"fact": fact, "gold": gold, "pred": pred, "passages": passages})
        print(f"\rProcessed {i}", end="", flush=True)

    result = {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives_first_10": false_positives[:10],
        "false_negatives_first_10": false_negatives[:10],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {args.output}")
    print("FP:", len(false_positives), "FN:", len(false_negatives))


if __name__ == "__main__":
    main()
