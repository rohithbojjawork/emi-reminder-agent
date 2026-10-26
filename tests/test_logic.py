import json
from datetime import date

import numpy as np

from agent.calllog import CallLog, percentile, summarize
from agent.language import LanguageTracker
from agent.flow import CallFlow
from agent.prompts import GREETING, interruption_note
from agent.ptp import validate_promise
from agent.record import identity_result, name_matches
from agent.speech import amount_words, date_words, number_words
from agent.vm_classifier import VoicemailClassifier, normalize
from agent.voicemail import VoicemailDetector, is_tone

TODAY = date(2026, 10, 24)


def test_amounts_and_dates():
    assert amount_words(4500, 'en') == 'four thousand five hundred rupees'
    assert amount_words(4500, 'hi') == 'चार हज़ार पांच सौ रुपये'
    assert number_words(120_000, 'en') == 'one lakh twenty thousand'
    assert number_words(25_00_00_000, 'hi') == 'पच्चीस करोड़'
    assert number_words(101, 'en') == 'one hundred one'
    assert number_words(38, 'hi') == 'अड़तीस'
    assert date_words(date(2026, 10, 10), 'en') == 'the tenth of October'
    assert date_words(date(2026, 10, 22), 'en') == 'the twenty second of October'
    assert date_words(date(2026, 10, 30), 'en') == 'the thirtieth of October'
    assert date_words(date(2026, 11, 3), 'en', weekday=True) == 'Tuesday, the third of November'
    assert date_words(date(2026, 10, 10), 'hi') == 'दस अक्टूबर'
    assert date_words(date(2026, 10, 31), 'hi', weekday=True) == 'शनिवार, इकतीस अक्टूबर'


def test_promise_validation():
    ok = validate_promise('2026-10-31', TODAY, 4500, 'LN-88213')
    assert ok['ok'] and ok['date_iso'] == '2026-10-31' and ok['reference_id'].startswith('PTP-')
    assert ok['confirmation']['en'] == ('I have noted your promise to pay four thousand five hundred rupees '
                                        'by Saturday, the thirty first of October. Is that right?')
    assert validate_promise('2026-10-24', TODAY, 4500, 'L')['error'] == 'date_not_in_future'
    assert validate_promise('2026-11-23', TODAY, 4500, 'L')['ok']  # exactly 30 days
    assert validate_promise('2026-11-24', TODAY, 4500, 'L')['error'] == 'date_too_far'
    assert validate_promise('next friday', TODAY, 4500, 'L')['error'] == 'invalid_date'


def test_identity():
    for said in ('Rahul', 'haan ji, Rahul bol raha hoon', 'राहुल शर्मा', 'my name is rahul sharma', 'Raahul'):
        assert name_matches(said), said
    for said in ('Amit', 'who is this?', '', 'wrong number'):
        assert not name_matches(said), said


def det(lang, source='llm', **kw):
    return {'lang': lang, 'source': source, **kw}


def test_language_resolves_per_turn():
    lt = LanguageTracker('hi')
    assert lt.observe(1, det(None, 'skipped', skip_reason='skip_phrase'), 2) is None  # "haan ji": no evidence
    assert lt.observe(2, det('en'), 2) is None                      # 2-word English turn: not confident
    assert lt.observe(3, det('en'), 5) == 'en'                      # confident English turn: this turn is English
    assert lt.observe(4, det('en'), 6) is None                      # same language: stays
    assert lt.observe(5, det('hi', 'heuristic'), 6) == 'hi'         # back to Hindi on the turn it happens
    assert lt.observe(6, det('en', explicit=True, keyword='talk in english'), 2) == 'en'  # explicit, even short
    assert [t['lang'] for t in lt.timeline] == ['hi', 'en', 'hi', 'en']
    assert [d['resolved'] for d in lt.decisions] == ['hi', 'hi', 'en', 'en', 'hi', 'en']
    assert all('reason' in d for d in lt.decisions)


def test_hinglish_fillers_do_not_flip():
    lt = LanguageTracker('en')  # short code-mixed replies and fillers carry no confident evidence
    for i, (lang, words) in enumerate([('hi', 2), (None, 1), ('hi', 1), (None, 3), ('hi', 2)], 1):
        lt.observe(i, det(lang, 'skipped' if lang is None else 'llm'), words)
    assert lt.lang == 'en' and len(lt.timeline) == 1


def test_explicit_mode_only_switches_on_request():
    lt = LanguageTracker('hi', mode='explicit')
    for i in range(1, 4):
        assert lt.observe(i, det('en', 'heuristic'), 8) is None
    assert lt.observe(4, det('en', explicit=True, keyword='english me'), 3) == 'en'


def test_voicemail_rules():
    human = VoicemailDetector()
    human.speech_start(0.6)
    human.speech_stop(1.4)
    assert human.tick(1.9) is None
    assert human.tick(2.3)['kind'] == 'human'

    machine = VoicemailDetector()
    machine.speech_start(0.6)
    machine.classified({'voicemail': True, 'score': 0.97, 'source': 'classifier'}, 2.6)
    assert machine.decision['kind'] == 'voicemail' and machine.decision['reason'] == 'classifier score 0.97'

    monologue = VoicemailDetector()  # long speech alone does not decide: the classifier does
    monologue.speech_start(0.5)
    assert monologue.tick(6.0) is None
    monologue.classified({'voicemail': False, 'score': 0.01, 'source': 'classifier'}, 6.1)
    assert monologue.tick(6.2) is None  # still talking
    monologue.speech_stop(6.3)
    assert monologue.tick(7.2)['reason'] == 'classifier said not voicemail and the callee paused'

    # the reported call: hi-IN's Devanagari final scores low and lands first; en-IN's lands 0.4 s later
    race = VoicemailDetector()
    race.speech_start(1.02)
    race.speech_stop(5.42)
    race.hold_until = 6.08 + 1.0  # the gate holds after one recogniser's final
    race.classified({'voicemail': False, 'score': 0.04, 'source': 'classifier'}, 6.08)
    assert race.tick(6.32) is None  # used to decide 'human' here
    race.classified({'voicemail': True, 'score': 1.0, 'source': 'classifier'}, 6.5)
    assert race.decision['kind'] == 'voicemail'

    silent = VoicemailDetector()
    assert silent.tick(2.9) is None
    assert silent.tick(3.0)['reason'].startswith('silent pickup')

    talker = VoicemailDetector()  # several short phrases with real pauses: undecided until the fail-safe
    for s in (0.5, 2.0, 3.5, 5.0, 6.5, 8.0):
        talker.speech_start(s)
        talker.speech_stop(s + 0.6)
        assert talker.tick(s + 0.9) is None
    assert talker.tick(10.0)['reason'].startswith('no voicemail evidence')


def test_voicemail_classifier():
    vm = VoicemailClassifier()
    for text in ('the number you have dialled is currently switched off please try again later',
                 'आप जिस व्यक्ति को कॉल कर रहे हैं वह अभी उपलब्ध नहीं है कृपया बीप के बाद अपना संदेश छोड़ें',
                 'आपके द्वारा डायल किया गया नंबर अभी व्यस्त है'):
        assert vm.classify([text])['voicemail'], text
    for text in ('hello who is this', 'हाँ जी बोलिए', 'haan ji Rahul bol raha hoon', 'मैंने पेमेंट कर दिया है'):
        assert not vm.classify([text])['voicemail'], text
    # Either recogniser can supply the stronger signal; blanks are ignored.
    r = vm.classify(['Aap jis number par call kar rahe hain',
                     'आप जिस नंबर पर कॉल कर रहे हैं वह अभी बंद है', ''])
    assert r['voicemail'] and r['source'] == 'classifier'
    assert normalize('नंबर  Busy, 9876!') == normalize('नंबर busy 1234')


def test_beep_detection():
    sr = 16000
    t = np.arange(int(0.1 * sr)) / sr
    beep = (8000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    noise = np.random.default_rng(0).normal(0, 3000, len(t)).astype(np.int16)
    assert is_tone(beep.tobytes(), sr)
    assert not is_tone(noise.tobytes(), sr)
    assert not is_tone((beep // 100).tobytes(), sr)  # too quiet


def test_latency_summary():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    turns = [{'stt_ms': v, 'llm_ttft_ms': v * 2, 'tts_ttfb_ms': None} for v in (100, 200, 300)]
    s = summarize(turns)
    assert s['stt_ms'] == {'n': 3, 'median': 200, 'p90': 300} and 'tts_ttfb_ms' not in s


def test_call_log_saves_transcript_tools_and_summary(tmp_path):
    log = CallLog('CUST-TEST', out_dir=tmp_path)
    log.mark_started()
    log.data['turns'].append({'turn': 1, 'stt_ms': 120, 'total_ms': 900})
    log.add_tool('verify_identity', {'answer': 'yes'}, {'verified': True})
    path = log.save(messages=[{'role': 'user', 'content': 'Yes, speaking.'},
                              {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Thank you.'}]}])
    call = json.loads(path.read_text())
    assert call['transcript'] == [{'role': 'user', 'text': 'Yes, speaking.'},
                                  {'role': 'assistant', 'text': 'Thank you.'}]
    assert call['tools'][0]['name'] == 'verify_identity'
    assert call['summary']['turn_count'] == 1 and call['summary']['tool_call_count'] == 1
    assert call['latency_summary']['stt_ms'] == {'n': 1, 'median': 120, 'p90': 120}


def test_interruption_note():
    note, missed = interruption_note('मैं आपकी EMI के बारे में कॉल कर रही हूँ। क्या आपने भुगतान कर दिया है?', 'मैं आपकी EMI के')
    assert missed == 'बारे में कॉल कर रही हूँ। क्या आपने भुगतान कर दिया है?'
    assert 'Played before the interruption: "मैं आपकी EMI के"' in note and 'Do not apologise' in note
    assert interruption_note('Hello there', '')[1] == 'Hello there'      # cut before any word was played
    assert interruption_note('Hello there', 'Hello there') == (None, '')  # nothing missed: no note
    # opening cut, even near its end: always re-introduce and re-ask who they are
    played = ' '.join(GREETING['hi'].split()[:14])
    note, _ = interruption_note(GREETING['hi'], played, greeting=True)
    assert 'AI assistant' in note and 'recorded' in note and 'speaking with the customer' in note


def test_identity_is_right_party_check():
    assert identity_result('yes', None, 'हाँ जी, बोलिए') == 'verified'
    assert identity_result('yes', None, 'Yes, speaking.') == 'verified'
    assert identity_result('unclear', None, 'हाँ, मैं राहुल बोल रहा हूँ') == 'verified'   # the name itself
    assert identity_result('no', 'रोहित', 'मेरा नाम रोहित है') == 'other_person'         # the reported call
    assert identity_result('yes', 'Rahul', 'My name is Rohit.') == 'unclear'   # model claims Rahul, transcript disagrees
    assert identity_result('yes', None, 'नहीं') == 'unclear'                    # "yes" without the caller's affirmation
    assert identity_result('unclear', None, 'OK, OK, I get it.') == 'unclear'


def test_call_flow_rules():
    flow = CallFlow('hi', TODAY)
    assert flow.end_call({'disposition': 'already_paid'})[0]['error'] == 'identity not verified yet'
    assert flow.repeat_reminder({})[0]['error'] == 'identity_not_verified'
    result, say = flow.verify_identity({'answer': 'no', 'name_said': 'रोहित'}, 'मेरा नाम रोहित है')
    assert result == {'verified': False, 'reason': 'other_person', 'attempts_left': 2} and say is None
    assert flow.end_call({'disposition': 'callback_requested'})[0]['error'] == 'ask for a callback time first'
    ok, goodbye, details = flow.end_call({'disposition': 'callback_requested', 'callback_time': 'शाम पाँच बजे'})
    assert ok == {'ok': True} and details == {'callback_time': 'शाम पाँच बजे'} and goodbye.startswith('ठीक है')
    result, say = flow.verify_identity({'answer': 'yes'}, 'हाँ बोलिए')
    assert result['verified'] and 'चार हज़ार पांच सौ रुपये' in say   # reminder spoken by the system
    assert flow.end_call({'disposition': 'promise_to_pay'})[0]['error'].startswith('no promise recorded')
    result, say = flow.record_promise_to_pay({'date': '2026-10-30'})
    assert result['ok'] and flow.end_call({'disposition': 'promise_to_pay'})[2]['promise_date'] == '2026-10-30'


def test_mid_call_lines_hand_the_turn_back():
    """Every line the system speaks mid-call ends with a question; only closings (then hang-up) don't."""
    from agent.prompts import reminder_text
    lines = [*GREETING.values(), reminder_text('en'), reminder_text('hi')]
    for d in ('2026-10-30', '2026-10-01', '2027-01-01', 'friday'):  # ok, past, too far, unparseable
        r = validate_promise(d, TODAY, 4500, 'L')
        lines += list((r['confirmation'] if r['ok'] else r['say_text']).values())
    assert [line for line in lines if not line.strip().endswith('?')] == []
