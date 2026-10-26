"""Promise-to-pay validation: runs server-side, the LLM only proposes a date."""
import hashlib
from datetime import date, timedelta

from agent.speech import amount_words, date_words

MAX_DAYS = 30


def validate_promise(date_iso: str, today: date, amount: int, loan_id: str) -> dict:
    """Structured confirmation or error for a proposed promise-to-pay date (ISO YYYY-MM-DD)."""
    earliest, latest = today + timedelta(days=1), today + timedelta(days=MAX_DAYS)
    allowed = {'earliest': earliest.isoformat(), 'latest': latest.isoformat()}
    window = {'en': f'any date from {date_words(earliest, "en")} to {date_words(latest, "en")}',
              'hi': f'{date_words(earliest, "hi")} से {date_words(latest, "hi")} के बीच की कोई तारीख़'}
    say = {  # pre-worded so the model never writes the allowed range itself
        'invalid_date': {'en': 'Sorry, I did not catch the date. Which date will you pay by?',
                         'hi': 'माफ़ कीजिए, मैं तारीख़ समझ नहीं पाई। आप किस तारीख़ तक भुगतान करेंगे?'},
        'date_not_in_future': {'en': f'That date has passed. Could you choose {window["en"]}?',
                               'hi': f'वह तारीख़ निकल चुकी है। क्या आप {window["hi"]} बता सकते हैं?'},
        'date_too_far': {'en': f'I can only note {window["en"]}. Could you pay a little earlier?',
                         'hi': f'मैं केवल {window["hi"]} नोट कर सकती हूँ। क्या आप थोड़ा पहले भुगतान कर सकते हैं?'},
    }
    error = None
    try:
        d = date.fromisoformat(str(date_iso).strip())
        if d <= today:
            error = 'date_not_in_future'
        elif d > latest:
            error = 'date_too_far'
    except ValueError:
        error = 'invalid_date'
    if error:
        return {'ok': False, 'error': error, 'allowed_range': allowed, 'say_text': say[error]}
    ref = 'PTP-' + hashlib.sha1(f'{loan_id}:{d}'.encode()).hexdigest()[:6].upper()
    return {
        'ok': True,
        'date_iso': d.isoformat(),
        'reference_id': ref,
        'confirmation': {
            'en': f'I have noted your promise to pay {amount_words(amount, "en")} by '
                  f'{date_words(d, "en", weekday=True)}. Is that right?',
            'hi': f'मैंने नोट कर लिया है कि आप {date_words(d, "hi", weekday=True)} तक '
                  f'{amount_words(amount, "hi")} जमा कर देंगे। क्या यह सही है?',
        },
    }
