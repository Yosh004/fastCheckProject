# factcheck.py

import gc
import re
from collections import Counter
from typing import List

import numpy as np
import spacy
import torch
from nltk.stem import PorterStemmer


# A small built-in stop-word list keeps the solution reproducible and avoids
# requiring an NLTK corpus download during marking.
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "of", "at", "by", "for", "from", "in", "into", "on", "onto", "to", "up",
    "down", "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now", "is",
    "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "this", "that", "these", "those",
    "as", "it", "its", "itself", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "they", "them", "their", "theirs", "themselves", "we",
    "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "i", "me", "my", "mine", "myself", "who", "whom", "which",
    "what", "also"
}

_STEMMER = PorterStemmer()


def _tokenize_content_words(text: str) -> List[str]:
    """Lowercase, tokenize, remove stop words, and Porter-stem a string."""
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return [_STEMMER.stem(tok) for tok in tokens if tok not in _STOP_WORDS]


def _word_recall(fact: str, evidence: str) -> float:
    """
    Fraction of the fact's content tokens that are found in the evidence.

    Counter intersection is used instead of just sets so repeated tokens are
    handled correctly.  This score is asymmetric on purpose: a long Wikipedia
    passage should not be penalised merely because it contains many extra words.
    """
    fact_tokens = _tokenize_content_words(fact)
    if not fact_tokens:
        return 0.0

    evidence_tokens = _tokenize_content_words(evidence)
    if not evidence_tokens:
        return 0.0

    fact_counts = Counter(fact_tokens)
    evidence_counts = Counter(evidence_tokens)
    overlap = sum((fact_counts & evidence_counts).values())
    return overlap / sum(fact_counts.values())


def _clean_passage_text(text: str) -> str:
    """Remove FActScore sentence tags and normalise whitespace."""
    text = re.sub(r"</?s>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_passage_into_sentences(text: str) -> List[str]:
    """
    Split a retrieved FActScore passage into sentence-like NLI premises.

    The data already contains <s>...</s> boundaries.  We use those first, then
    apply a light regex sentence split.  This avoids needing the NLTK Punkt
    resource and keeps the coursework self-contained.
    """
    tagged_chunks = [
        chunk.strip()
        for chunk in re.split(r"</?s>|\n+", text)
        if chunk.strip()
    ]

    sentences = []
    for chunk in tagged_chunks:
        chunk = _clean_passage_text(chunk)
        if not chunk:
            continue
        # Conservative split: punctuation followed by whitespace and a likely
        # sentence-start character.  If it does not split, the whole chunk is kept.
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', chunk)
        sentences.extend(p.strip() for p in parts if p.strip())

    if not sentences:
        cleaned = _clean_passage_text(text)
        return [cleaned] if cleaned else []
    return sentences


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.model.eval()

    def _entailment_index(self) -> int:
        """Find the entailment class robustly; fall back to index 0 for this model."""
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, label in id2label.items():
            if str(label).lower() == "entailment":
                return int(idx)

        label2id = getattr(self.model.config, "label2id", None) or {}
        for label, idx in label2id.items():
            if str(label).lower() == "entailment":
                return int(idx)

        # The coursework comments specify the order
        # ["entailment", "neutral", "contradiction"].
        return 0

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """Return P(entailment) for a premise/hypothesis pair."""
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            entailment_idx = self._entailment_index()
            entailment_probability = probabilities[0, entailment_idx].item()

        # Explicit cleanup matters when hundreds/thousands of NLI calls are made.
        del inputs, outputs, logits, probabilities
        if self.cuda:
            torch.cuda.empty_cache()
        gc.collect()

        return entailment_probability


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """
    Bag-of-words classifier using the maximum content-word recall across
    retrieved passages.

    The threshold was tuned on the supplied development set, as permitted by
    the coursework brief.  With the supplied data it gives about 82.35%
    accuracy (IR is mapped to NS by the provided driver).
    """

    THRESHOLD = 0.6333333333333333

    @staticmethod
    def overlap_score(fact: str, passage_text: str) -> float:
        return _word_recall(fact, passage_text)

    def predict(self, fact: str, passages: List[dict]) -> str:
        if not passages:
            return "NS"

        best_score = max(
            (self.overlap_score(fact, passage.get("text", "")) for passage in passages),
            default=0.0,
        )

        return "S" if best_score >= self.THRESHOLD else "NS"


class EntailmentFactChecker(FactChecker):
    """
    Textual-entailment fact checker.

    1. Split retrieved passages into sentence-like premises.
    2. Rank premises by inexpensive lexical overlap.
    3. Prune very weak candidates and cap the number of NLI calls.
    4. Return S if any retained premise has sufficiently high P(entailment).
    """

    # A conventional probability cutoff.  The brief permits tuning this value.
    ENTAILMENT_THRESHOLD = 0.50

    # Low lexical threshold used only for efficient pruning.  We still keep the
    # strongest few candidates even if none reaches this level.
    MIN_OVERLAP_FOR_NLI = 0.10
    MAX_NLI_CANDIDATES = 12
    FALLBACK_CANDIDATES = 3

    def __init__(self, ent_model):
        self.ent_model = ent_model

    def _candidate_sentences(self, fact: str, passages: List[dict]) -> List[str]:
        scored = []
        seen = set()

        for passage in passages:
            for sentence in _split_passage_into_sentences(passage.get("text", "")):
                if len(sentence) < 8:
                    continue
                # Avoid duplicate NLI calls caused by overlapping retrieved chunks.
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                scored.append((_word_recall(fact, sentence), sentence))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        retained = [item for item in scored if item[0] >= self.MIN_OVERLAP_FOR_NLI]

        # If a paraphrase has unusually low surface overlap, do not discard all
        # evidence: keep a small number of the best candidates as a safety net.
        if not retained:
            retained = scored[:self.FALLBACK_CANDIDATES]

        return [sentence for _, sentence in retained[:self.MAX_NLI_CANDIDATES]]

    def predict(self, fact: str, passages: List[dict]) -> str:
        candidates = self._candidate_sentences(fact, passages)
        if not candidates:
            return "NS"

        best_entailment = 0.0
        for premise in candidates:
            score = self.ent_model.check_entailment(premise, fact)
            best_entailment = max(best_entailment, score)

            # Early exit saves substantial inference time.
            if best_entailment >= self.ENTAILMENT_THRESHOLD:
                return "S"

        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    """Optional extension based on overlap of semantically useful dependency arcs."""

    THRESHOLD = 0.50

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_dependencies = self.get_dependencies(fact)
        if not fact_dependencies:
            return "NS"

        best_score = 0.0
        for passage in passages:
            passage_dependencies = self.get_dependencies(_clean_passage_text(passage.get("text", "")))
            if passage_dependencies:
                score = len(fact_dependencies & passage_dependencies) / len(fact_dependencies)
                best_score = max(best_score, score)

        return "S" if best_score >= self.THRESHOLD else "NS"

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
