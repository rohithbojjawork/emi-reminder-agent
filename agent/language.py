"""Per-turn language resolution on top of LanguageDetector (agent/lid.py).

Each turn resolves to one language; that recogniser's transcript goes to the LLM and the reply is in
that language. Code-mixed (Hinglish) turns don't flip it, because of a confidence threshold:
  - explicit requests ("hindi me baat karo", "talk in English") switch at once, in any mode
  - otherwise a turn needs MIN_SWITCH_WORDS+ words and a confident detection (heuristic or LLM)
  - fillers, numbers and short turns carry no evidence (gated in lid.py); LANG_PROMPT maps Hinglish to Hindi
"""
ROMAN_HINDI = set('''
haan han ha nahi nahin na ji kal parso aaj abhi paise paisa rupaye rupay kar karunga karungi kardunga
dunga dungi denge de diya diye hai hain tha thi hoon hu main mai mera meri mujhe aap aapka aapko hum
humne kya kab kaise kyun kyon theek thik accha acha baat bolo boliye bol raha rahi ho gaya gayi
bhai sahab wala wali tak se ko ka ki ke mein bhi lekin par phir jama bharunga bharungi
'''.split())

MIN_SWITCH_WORDS = 3


class LanguageTracker:
    def __init__(self, start_lang, mode='implicit'):
        self.lang, self.mode = start_lang, mode
        self.timeline = [{'turn': 0, 'lang': start_lang, 'reason': 'preferred_language from customer record'}]
        self.decisions = []

    def observe(self, turn, detection, words):
        """Apply one turn's detection; returns the new language if it switched, else None."""
        other = 'en' if self.lang == 'hi' else 'hi'
        detected = detection.get('lang')
        if detection.get('explicit') and detected == other:
            decision, reason = 'switch', f'explicit request (keyword "{detection.get("keyword")}")'
        elif detected is None:
            decision, reason = 'stay', f'no evidence ({detection.get("skip_reason") or detection.get("source")})'
        elif detected == self.lang:
            decision, reason = 'stay', f'detected {detected} = current'
        elif self.mode == 'explicit':
            decision, reason = 'stay', f'detected {detected}, but explicit mode switches only on request'
        elif words < MIN_SWITCH_WORDS:
            decision, reason = 'stay', f'detected {detected} on a {words}-word turn (< {MIN_SWITCH_WORDS}): not confident'
        else:
            decision, reason = 'switch', f'detected {detected} ({detection.get("source")}) on a {words}-word turn'
        entry = {'turn': turn, 'current': self.lang, 'decision': decision, 'reason': reason, 'words': words,
                 **{k: detection.get(k) for k in ('lang', 'source', 'keyword', 'skip_reason', 'heuristic',
                                                  'llm_answer', 'lid_ms', 'total_results') if detection.get(k) is not None}}
        if decision == 'switch':
            self.lang = other
            self.timeline.append({'turn': turn, 'lang': other, 'reason': reason})
        entry['resolved'] = self.lang
        self.decisions.append(entry)
        return self.lang if decision == 'switch' else None
