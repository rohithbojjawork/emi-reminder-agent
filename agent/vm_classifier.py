"""Character n-gram voicemail classifier with bundled weights and a small pure-Python scorer."""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

MODEL_PATH = Path(__file__).with_name('voicemail_model.json')
NGRAMS = (2, 5)


def normalize(text):
    """Fold common Unicode variants and punctuation while preserving Hindi and Latin script."""
    text = unicodedata.normalize('NFC', text.lower()).replace('़', '').replace('ँ', 'ं')
    text = re.sub(r'\d', '0', text)
    return ' '.join(re.sub(r'[^\wऀ-ॿ]+', ' ', text).split())


def ngrams(text):
    text = f' {normalize(text)} '
    return [text[index:index + size] for size in range(NGRAMS[0], NGRAMS[1] + 1)
            for index in range(len(text) - size + 1)]


class VoicemailClassifier:
    """Score transcripts from both recognizers; ambiguous or unseen text defaults to human."""

    def __init__(self, path=MODEL_PATH):
        model = json.loads(Path(path).read_text())
        self.grams, self.intercept, self.threshold = model['grams'], model['intercept'], model['threshold']

    def score(self, text):
        counts = Counter(gram for gram in ngrams(text) if gram in self.grams)
        if not counts:
            return 1 / (1 + math.exp(-self.intercept))
        weights = {gram: (1 + math.log(count)) * self.grams[gram][0] for gram, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        logit = self.intercept + sum(value * self.grams[gram][1] for gram, value in weights.items()) / norm
        return 1 / (1 + math.exp(-logit))

    def classify(self, texts):
        score = max((self.score(text) for text in texts if text.strip()), default=0.0)
        return {'voicemail': score >= self.threshold, 'score': round(score, 3), 'source': 'classifier'}