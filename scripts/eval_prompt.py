"""Prompt eval: scripted conversations against the real LLM with the bot's prompt, tool schemas and tool rules
(agent/flow.CallFlow), mirroring how bot.py builds the context (scripted speech appended as assistant messages,
interruption and language-switch notes as system messages). Each scenario runs --runs times; checks are rules on
what the agent said and which tools it called.

  uv run python scripts/eval_prompt.py [--runs 3] [--only name,...] [--show]
"""
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import TOOLS  # noqa: E402
from agent.flow import CallFlow  # noqa: E402
from agent.prompts import GREETING, interruption_note, language_switch_note, reminder_text, system_prompt  # noqa: E402

TODAY = date(2026, 10, 24)  # a Saturday: "next Friday" is 2026-10-30
LOAN_WORDS = re.compile(r'EMI|लोन|loan|रुपये|rupees|बकाया|overdue|pending|due|किस्त|भुगतान|payment', re.I)
ASKS_NAME = re.compile(r'आपका नाम|अपना नाम|your name', re.I)
APOLOGY = re.compile(r'apolog|sorry|क्षमा|माफ़|माफ', re.I)
DEVANAGARI = re.compile(r'[ऀ-ॿ]')


def tools_openai():
    return [{'type': 'function', 'function': {'name': f.name, 'description': f.description, 'parameters': {
        'type': 'object', 'properties': f.properties, 'required': f.required}}} for f in TOOLS.standard_tools]


# ---- scenario building blocks ----
def cut_greeting(lang, words):
    """The greeting cut after `words` words, plus the note the bot adds."""
    played = ' '.join(GREETING[lang].split()[:words])
    return [{'role': 'assistant', 'content': played},
            {'role': 'system', 'content': interruption_note(GREETING[lang], played, greeting=True)[0]}]


def greeted(lang):
    return [{'role': 'assistant', 'content': GREETING[lang]}]


def verified(lang):
    yes = {'hi': 'हाँ जी, बोलिए', 'en': 'Yes, speaking.'}[lang]
    return greeted(lang) + [{'role': 'user', 'content': yes}, {'role': 'assistant', 'content': reminder_text(lang)}]


def S(name, lang, prefix, turns, checks, switch_to=None, pre_verified=False):
    return dict(name=name, lang=lang, prefix=prefix, turns=turns, checks=checks, switch_to=switch_to,
                pre_verified=pre_verified)


# ---- checks: run(r) -> True/False; r has .said (agent's own text), .calls, .disposition, .verified ----
def says(rx):
    return (f'says /{rx}/', lambda r: bool(re.search(rx, r['last_said'], re.I)))


def not_says(rx, label=None):
    return (label or f'not /{rx}/', lambda r: not re.search(rx, r['all_said'], re.I))


def called(tool, **kw):
    return (f'called {tool} {kw or ""}'.strip(), lambda r: any(
        c['name'] == tool and all(c['args'].get(k) == v for k, v in kw.items()) for c in r['calls']))


def not_called(tool):
    return (f'no {tool}', lambda r: not any(c['name'] == tool for c in r['calls']))


def ended(d):
    return (f'end_call {d}', lambda r: r['disposition'] == d)


def not_ended():
    return ('call not ended', lambda r: r['disposition'] is None)


def is_verified(v=True):
    return (f'verified={v}', lambda r: r['verified'] is v)


no_loan = not_says(LOAN_WORDS.pattern, 'no loan details')
in_english = ('reply in English', lambda r: r['last_said'] and not DEVANAGARI.search(r['last_said']))
in_hindi = ('reply in Hindi', lambda r: bool(DEVANAGARI.search(r['last_said'])))
no_apology = not_says(APOLOGY.pattern, 'no apology')
reintro_en = ('re-introduces (Asha, AI, recorded)', lambda r: all(re.search(x, r['last_said'], re.I) for x in ('asha', r'\bAI\b', 'record')))
reintro_hi = ('re-introduces (आशा, AI, रिकॉर्ड)', lambda r: all(re.search(x, r['last_said']) for x in ('आशा', 'AI', 'रिकॉर्ड')))

SCENARIOS = [
    # the reported call: greeting cut near the end, caller answers in English
    S('greeting_cut_switch_en', 'hi', cut_greeting('hi', 14), ['OK, OK, I get it.'],
      [in_english, reintro_en, says('rahul'), no_apology, no_loan, not_ended()], switch_to='en'),
    S('greeting_cut_early_hi', 'hi', cut_greeting('hi', 3), ['हेलो, कौन बोल रहा है?'],
      [in_hindi, reintro_hi, says('राहुल'), no_apology, no_loan, not_ended()]),
    S('other_name_hi', 'hi', greeted('hi'), ['मेरा नाम रोहित है।'],
      [says('राहुल'), not_says(ASKS_NAME.pattern, 'does not ask their name'), no_loan, is_verified(False), not_ended()]),
    S('other_name_en', 'en', greeted('en'), ['My name is Rohit.'],
      [says('rahul'), not_says(ASKS_NAME.pattern, 'does not ask their name'), no_loan, is_verified(False), not_ended()]),
    S('third_party_unavailable', 'hi', greeted('hi'), ['मेरा नाम रोहित है।', 'राहुल मेरे भाई हैं, अभी घर पर नहीं हैं।'],
      [says('कब|समय|टाइम|when|time'), no_loan, not_ended()]),
    S('third_party_callback', 'hi', greeted('hi'),
      ['मेरा नाम रोहित है।', 'राहुल मेरे भाई हैं, अभी घर पर नहीं हैं।', 'शाम पाँच बजे के बाद कॉल कर लेना।'],
      [ended('callback_requested'), no_loan]),
    S('third_party_no_time', 'hi', greeted('hi'),
      ['नहीं, मैं उनकी पत्नी बोल रही हूँ।', 'वो बाहर गए हैं, पता नहीं कब आएँगे।', 'नहीं पता, आप बाद में देख लेना।'],
      [ended('customer_unavailable'), no_loan]),
    S('handover', 'hi', greeted('hi'), ['मेरा नाम रोहित है।', 'रुकिए, देता हूँ... हाँ, मैं राहुल बोल रहा हूँ।'],
      [is_verified(True)]),
    S('wrong_number', 'hi', greeted('hi'), ['यहाँ कोई राहुल नहीं रहता, गलत नंबर है।'], [ended('wrong_number'), no_loan]),
    S('who_are_you', 'hi', greeted('hi'), ['आप कौन बोल रहे हैं?'], [says('आशा'), says('राहुल'), no_loan, not_ended()]),
    S('rpc_yes_hi', 'hi', greeted('hi'), ['हाँ जी, बोलिए।'], [called('verify_identity', answer='yes'), is_verified(True)]),
    S('rpc_yes_en', 'en', greeted('en'), ['Yes, speaking.'], [called('verify_identity', answer='yes'), is_verified(True)]),
    S('busy_then_time', 'hi', greeted('hi'), ['अभी मैं गाड़ी चला रहा हूँ, बाद में कॉल करो।', 'कल सुबह ग्यारह बजे।'],
      [ended('callback_requested'), no_loan]),
    S('already_paid', 'hi', verified('hi'), ['मैंने तो परसों ही पेमेंट कर दिया था।'],
      [ended('already_paid'), not_says(r'रसीद|receipt|reference|रेफ़रेंस|ट्रांज़ैक्शन', 'no receipt ask')], pre_verified=True),
    S('ptp_next_friday', 'hi', verified('hi'), ['अगले शुक्रवार तक कर दूँगा।'],
      [called('record_promise_to_pay', date='2026-10-30')], pre_verified=True),
    S('ptp_confirm', 'hi', verified('hi'), ['अगले शुक्रवार तक कर दूँगा।', 'हाँ, सही है।'],
      [called('record_promise_to_pay', date='2026-10-30'), ended('promise_to_pay')], pre_verified=True),
    S('ptp_correct_date', 'hi', verified('hi'), ['अगले शुक्रवार तक कर दूँगा।', 'नहीं, शुक्रवार नहीं, परसों कर दूँगा।'],
      [called('record_promise_to_pay', date='2026-10-26'), not_ended()], pre_verified=True),
    S('ptp_vague', 'hi', verified('hi'), ['जल्दी कर दूँगा।'],
      [not_called('record_promise_to_pay'), says('तारीख|दिन|कब|date'), not_ended()], pre_verified=True),
    S('refuse_twice', 'hi', verified('hi'), ['मैं पैसे नहीं दूँगा।', 'नहीं, बिल्कुल नहीं दूँगा।'],
      [ended('refused_to_pay')], pre_verified=True),
    S('not_my_loan', 'hi', verified('hi'), ['मैंने कोई लोन नहीं लिया।'], [ended('loan_not_recognised')], pre_verified=True),
    S('do_not_call', 'hi', verified('hi'), ['मुझे दोबारा कॉल मत करना।'], [ended('do_not_call')], pre_verified=True),
    S('amount_again', 'hi', verified('hi'), ['कितने पैसे बाकी हैं?'], [called('repeat_reminder')], pre_verified=True),
    S('hardship', 'hi', verified('hi'), ['नौकरी चली गई है, अभी नहीं हो पाएगा।'],
      [says('तारीख|कब|दिन'), not_ended()], pre_verified=True),
]
GLOBAL = [('no digits in agent text', lambda r: not re.search(r'\d', r['all_said'])),
          ('every reply ends with a question', lambda r: all(t.strip().endswith('?') for t in r['said'])),
          ('no OTP/PIN ask', lambda r: not re.search(r'OTP|PIN|पासवर्ड|password', r['all_said'], re.I))]


async def run_scenario(client, model, sc, sem):
    flow = CallFlow(sc['lang'], TODAY)
    flow.verified = sc['pre_verified']
    messages = [{'role': 'system', 'content': system_prompt(sc['lang'], TODAY)}] + json.loads(json.dumps(sc['prefix']))
    r = {'calls': [], 'said': [], 'disposition': None, 'error': None}
    for i, turn in enumerate(sc['turns']):
        if i == 0 and sc['switch_to']:  # BilingualTranscriber switched on this turn: new prompt + switch note
            flow.lang = sc['switch_to']
            messages[0] = {'role': 'system', 'content': system_prompt(flow.lang, TODAY)}
            messages.append({'role': 'system', 'content': language_switch_note(flow.lang)})
        messages.append({'role': 'user', 'content': turn})
        for _ in range(4):  # model -> tools -> model, until it speaks or the system speaks for it
            async with sem:
                resp = await client.chat.completions.create(model=model, messages=messages, tools=tools_openai(),
                                                            temperature=0.2)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                messages.append({'role': 'assistant', 'content': msg.content or ''})
                r['said'].append(msg.content or '')
                break
            messages.append({'role': 'assistant', 'content': msg.content or '', 'tool_calls': [
                {'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                for c in msg.tool_calls]})
            spoken = None
            for c in msg.tool_calls:
                args = json.loads(c.function.arguments or '{}')
                r['calls'].append({'name': c.function.name, 'args': args})
                if c.function.name == 'end_call':
                    result, spoken, _ = flow.end_call(args)
                    if spoken:
                        r['disposition'] = args['disposition']
                elif c.function.name == 'verify_identity':
                    result, spoken = flow.verify_identity(args, turn)
                else:
                    result, spoken = getattr(flow, c.function.name)(args)
                messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': json.dumps(result, ensure_ascii=False)})
            if spoken:  # run_llm=False: the system's speech ends the turn
                messages.append({'role': 'assistant', 'content': spoken})
                break
        if r['disposition']:
            break
    r['verified'] = flow.verified
    r['all_said'] = ' '.join(r['said'])
    r['last_said'] = r['said'][-1] if r['said'] else ''
    r['results'] = {label: bool(fn(r)) for label, fn in sc['checks'] + GLOBAL}
    return r


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--only')
    ap.add_argument('--show', action='store_true', help='print what the agent said')
    ap.add_argument('--report')
    a = ap.parse_args()
    load_dotenv(override=True)
    key = (os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or '').strip().removeprefix('Bearer ').strip()
    client, model = AsyncOpenAI(api_key=key, base_url=os.getenv('LLM_BASE_URL') or None), os.getenv('LLM_MODEL')
    scenarios = [s for s in SCENARIOS if not a.only or s['name'] in a.only.split(',')]
    sem = asyncio.Semaphore(6)
    runs = await asyncio.gather(*[run_scenario(client, model, s, sem) for s in scenarios for _ in range(a.runs)])
    total_pass = checks_pass = checks_total = 0
    report = []
    for i, s in enumerate(scenarios):
        rs = runs[i * a.runs:(i + 1) * a.runs]
        passed = sum(all(r['results'].values()) for r in rs)
        total_pass += passed
        fails = sorted({k for r in rs for k, v in r['results'].items() if not v})
        checks_pass += sum(v for r in rs for v in r['results'].values())
        checks_total += sum(len(r['results']) for r in rs)
        print(f'{"PASS" if passed == a.runs else "FAIL"} {passed}/{a.runs}  {s["name"]:26s} {"; ".join(fails)}')
        if a.show or passed < a.runs:
            for r in rs:
                print(f'      said: {" | ".join(r["said"])[:220]}  calls: {[c["name"] + json.dumps(c["args"], ensure_ascii=False) for c in r["calls"]]}')
        report.append({'scenario': s['name'], 'passed': passed, 'runs': a.runs, 'failed_checks': fails,
                       'transcripts': [{'said': r['said'], 'calls': r['calls'], 'disposition': r['disposition']} for r in rs]})
    n = len(scenarios) * a.runs
    print(f'\nscenario runs passed: {total_pass}/{n} ({100 * total_pass / n:.0f}%), checks passed: {checks_pass}/{checks_total}, model: {model}')
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    asyncio.run(main())
