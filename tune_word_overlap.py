"""Development-set helper for explaining/tuning the Part 1 threshold.
Not required for submission.
"""
import json
from factcheck import WordRecallThresholdFactChecker

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"


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


def accuracy_at_threshold(examples, threshold):
    correct = 0
    for fact, gold, passages in examples:
        best = max(
            WordRecallThresholdFactChecker.overlap_score(fact, p.get("text", ""))
            for p in passages
        )
        pred = "S" if best >= threshold else "NS"
        correct += int(pred == gold)
    return correct / len(examples)


if __name__ == "__main__":
    examples = read_data()
    best = (0.0, None)
    print("threshold\taccuracy")
    for i in range(0, 101):
        threshold = i / 100
        acc = accuracy_at_threshold(examples, threshold)
        print(f"{threshold:.2f}\t\t{acc:.4f}")
        if acc > best[0]:
            best = (acc, threshold)

    print(f"\nBest grid threshold: {best[1]:.2f}")
    print(f"Best grid accuracy:  {best[0]:.4f}")
    print("Final factcheck.py uses a finer tuned threshold near 0.6333.")
