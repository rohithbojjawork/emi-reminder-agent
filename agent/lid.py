"""Hindi/English language identification with transcript gates, heuristic detection, and an LLM fallback."""
import asyncio
import re
import time

from loguru import logger

from agent.language import ROMAN_HINDI
from agent.lid_data import EXPLICIT_LANGUAGE_KEYWORDS, LANG_PROMPT, NUMBER_WORDS, SKIP_PHRASES

LANG_NAMES = {'hi': 'Hindi', 'en': 'English'}
STT_CODES = {'hi': 'hi-IN', 'en': 'en-IN'}
TIMEOUTS_S = (0.8, 1.2)
_NONE = {'none', 'null', 'nil', 'na', 'n/a', 'no change', 'same', 'unchanged'}

_PUNCT = str.maketrans('', '', '.,!?;:()[]{}"\'-')
_KW_FROM = '.,!?;:()[]{}"\'-/@#*_\\'
_KW_STRIP = str.maketrans(_KW_FROM, ' ' * len(_KW_FROM))


def _clean(s):
    return ' '.join(s.lower().translate(_PUNCT).split())


_SKIP = frozenset(c for c in map(_clean, SKIP_PHRASES) if c)
_NUMS = frozenset(c for c in map(_clean, NUMBER_WORDS) if c)
_CUES = tuple(sorted({' '.join(k.lower().translate(_KW_STRIP).split()) for k in EXPLICIT_LANGUAGE_KEYWORDS} - {''},
                     key=lambda c: -len(c)))


def gate(total, prev_lang, mode, min_words):
    """(skip_reason or None, keyword or None) — same order as our detect_language gates."""
    texts = [t.strip() for t in total.values() if t and t.strip()]
    if not texts:
        return 'no_transcript', None
    first = _clean(texts[0])
    if first and all(w in _NUMS for w in first.split()):
        return 'numeric', None
    for t in texts:
        n = _clean(t)
        words = n.split()
        if n in _SKIP or (len(words) >= 2 and len(set(words)) == 1 and words[0] in _SKIP):
            return 'skip_phrase', None
    combined = ' '.join(' '.join(texts).lower().translate(_KW_STRIP).split())
    keyword = next((c for c in _CUES if c in combined), None)
    if keyword:
        return None, keyword
    if mode == 'explicit':
        return 'explicit_mode_no_keyword', None
    if max(len(t.split()) for t in texts) < min_words:
        return 'min_words', None
    if not (total.get(STT_CODES[prev_lang]) or '').strip():
        return 'prev_lang_null', None
    return None, None


def heuristic(total):
    """Clear-cut cases only, else None. Hindi speech through the English recogniser comes out as
    romanised Hindi; English speech through the Hindi recogniser keeps Latin-script English words."""
    en, hi = total.get('en-IN') or '', total.get('hi-IN') or ''
    en_words = re.findall(r'[a-z]+', en.lower())
    roman = sum(w in ROMAN_HINDI for w in en_words) / len(en_words) if en_words else 0.0
    letters = re.findall(r'[A-Za-zऀ-ॿ]', hi)
    latin_in_hi = sum(c.isascii() for c in letters) / len(letters) if letters else 0.0
    signals = {'en_stt_roman_hindi': round(roman, 2), 'hi_stt_latin': round(latin_in_hi, 2)}
    if roman >= 0.4 and latin_in_hi < 0.3:
        return 'hi', signals
    if roman == 0 and len(en_words) >= 3 and latin_in_hi >= 0.5:
        return 'en', signals
    return None, signals


def build_user_text(total, history):
    """Same layout as ours (_llm_detect): last 4 turns, then one line per recogniser."""
    lines = []
    if history:
        lines.append('Conversation History:')
        for role, lang, text in history[-4:]:
            lines.append(f'Previous User ({LANG_NAMES.get(lang, "unknown")}): {text}' if role == 'user'
                         else f'Previous Bot: {text}')
        lines.append('\nCurrent User Input:')
    lines += [f'{LANG_NAMES[l]} STT: {total[c]}' for l, c in STT_CODES.items() if (total.get(c) or '').strip()]
    return '\n'.join(lines)


def parse(answer):
    a = (answer or '').strip().strip('"').lower()
    if not a or a in _NONE:
        return None
    return next((l for l, name in LANG_NAMES.items() if name.lower() in a), None)


class LanguageDetector:
    """detect() returns {'lang': 'hi'|'en'|None, 'explicit': bool, 'source': ..., ...}; None = no evidence."""

    def __init__(self, client=None, model='gpt-4.1-mini', mode='implicit', prompt=LANG_PROMPT, min_words=2):
        self.client, self.model, self.mode, self.prompt, self.min_words = client, model, mode, prompt, min_words

    async def detect(self, total, prev_lang, history=()):
        started = time.monotonic()
        skip, keyword = gate(total, prev_lang, self.mode, self.min_words)
        out = {'lang': None, 'explicit': bool(keyword), 'keyword': keyword, 'skip_reason': skip,
               'source': 'skipped' if skip else None, 'total_results': total}
        if skip:
            return out
        if not keyword:  # clear-cut turns skip the LLM; explicit requests always ask it for the target
            lang, out['heuristic'] = heuristic(total)
            if lang:
                out.update(lang=lang, source='heuristic')
                return out
        out.update(await self._llm(total, history))
        out['lid_ms'] = round((time.monotonic() - started) * 1000)
        return out

    async def _llm(self, total, history):
        if self.client is None:
            return {'lang': None, 'source': 'llm_unavailable'}
        messages = [{'role': 'system', 'content': self.prompt},
                    {'role': 'user', 'content': build_user_text(total, history)}]
        for attempt, timeout in enumerate(TIMEOUTS_S, 1):
            try:
                r = await asyncio.wait_for(self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.1, max_tokens=20), timeout)
                answer = r.choices[0].message.content
                return {'lang': parse(answer), 'source': 'llm', 'llm_answer': answer, 'attempts': attempt}
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f'LanguageDetector LLM error: {e}')
                return {'lang': None, 'source': 'llm_error', 'attempts': attempt}
        return {'lang': None, 'source': 'llm_timeout', 'attempts': len(TIMEOUTS_S)}  # ours: fall back to prev_lang
