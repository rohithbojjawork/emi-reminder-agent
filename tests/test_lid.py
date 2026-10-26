import asyncio
from types import SimpleNamespace

from agent.lid import LanguageDetector, build_user_text, gate, heuristic, parse


def tr(hi, en):
    return {'hi-IN': hi, 'en-IN': en}


def test_gates_in_our_order():
    assert gate(tr('', ''), 'hi', 'implicit', 2) == ('no_transcript', None)
    assert gate(tr('हाँ जी', 'haan ji'), 'hi', 'implicit', 2)[0] == 'skip_phrase'
    assert gate(tr('ok ok', 'ok ok'), 'hi', 'implicit', 2)[0] == 'skip_phrase'          # repeated filler
    assert gate(tr('चार पांच', 'four five'), 'hi', 'implicit', 2)[0] == 'numeric'
    # our NUMBER_WORDS_ALL_LANGUAGES has no "thousand"/"hundred", so amounts go on to language ID (same as prod)
    assert gate(tr('फोर थाउजेंड', 'four thousand'), 'hi', 'implicit', 2)[0] != 'numeric'
    assert gate(tr('हिंदी में बात कीजिए', 'hindi me baat kijiye'), 'en', 'explicit', 2)[1]  # keyword cue
    assert gate(tr('आई विल पे टुमारो', 'I will pay tomorrow'), 'hi', 'explicit', 2)[0] == 'explicit_mode_no_keyword'
    assert gate(tr('', 'tomorrow'), 'en', 'implicit', 2)[0] == 'skip_phrase'  # "tomorrow" is in our skip list
    assert gate(tr('', 'salary'), 'en', 'implicit', 2)[0] == 'min_words'
    assert gate(tr('', 'I will pay it tomorrow'), 'hi', 'implicit', 2)[0] == 'prev_lang_null'
    assert gate(tr('आई विल पे इट टुमारो', 'I will pay it tomorrow'), 'hi', 'implicit', 2) == (None, None)


def test_heuristic_settles_only_clear_cases():
    assert heuristic(tr('मैं कल पैसे जमा कर दूंगा', 'main kal paise jama kar dunga'))[0] == 'hi'
    assert heuristic(tr('I will pay it tomorrow', 'I will pay it tomorrow'))[0] == 'en'
    assert heuristic(tr('आई विल पे इट टुमारो', 'I will pay it tomorrow'))[0] is None  # needs the LLM


def test_parse_and_prompt_input():
    assert parse('Hindi') == 'hi' and parse('"English"') == 'en' and parse('None') is None and parse('same') is None
    text = build_user_text(tr('आई विल पे', 'I will pay'), [('assistant', None, 'Namaste'), ('user', 'hi', 'haan')])
    assert text.splitlines() == ['Conversation History:', 'Previous Bot: Namaste', 'Previous User (Hindi): haan', '',
                                 'Current User Input:', 'Hindi STT: आई विल पे', 'English STT: I will pay']


class FakeClient:
    def __init__(self, answer, delay=0.0):
        self.calls = 0

        async def create(**kwargs):
            self.calls += 1
            await asyncio.sleep(delay)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_detector_tiers_and_timeout_ladder(monkeypatch):
    ambiguous = tr('आई विल पे इट टुमारो', 'I will pay it tomorrow')
    client = FakeClient('English')
    d = await LanguageDetector(client).detect(ambiguous, 'hi')
    assert (d['lang'], d['source'], client.calls) == ('en', 'llm', 1)

    clear = tr('मैं कल पैसे जमा कर दूंगा', 'main kal paise jama kar dunga')
    d = await LanguageDetector(client).detect(clear, 'en')
    assert (d['lang'], d['source'], client.calls) == ('hi', 'heuristic', 1)  # no LLM call

    d = await LanguageDetector(client).detect(tr('इंग्लिश में बात कीजिए please', 'English mein baat kijiye please'), 'hi')
    assert d['explicit'] and d['source'] == 'llm' and client.calls == 2  # explicit request: LLM picks target

    monkeypatch.setattr('agent.lid.TIMEOUTS_S', (0.01, 0.01))
    slow = FakeClient('English', delay=0.2)
    d = await LanguageDetector(slow).detect(ambiguous, 'hi')
    assert (d['lang'], d['source'], slow.calls) == (None, 'llm_timeout', 2)  # falls back to current language

    # Error handling: if the LLM client raises an exception (e.g. 401 or 500), fall back gracefully
    class ErrorClient:
        def __init__(self):
            async def create(**kwargs):
                raise RuntimeError("API error")
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    d_err = await LanguageDetector(ErrorClient()).detect(ambiguous, 'hi')
    assert d_err['lang'] is None and d_err['source'] == 'llm_error'
