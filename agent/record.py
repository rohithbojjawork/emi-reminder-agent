"""The hard-coded (synthetic) customer record and identity matching."""
import difflib
import re
import unicodedata
from datetime import date

CUSTOMER = {
    'customer_id': 'CUST-1042',
    'name': 'Rahul Sharma',
    'name_hi': 'राहुल शर्मा',
    'phone': '+91XXXXXXXXXX',
    'loan_id': 'LN-88213',
    'emi_amount_inr': 4500,
    'due_date': date(2026, 10, 10),
    'preferred_language': 'hi',
    'lender': 'Sahayak Finance',  # fictional lender
}

# Romanised and Devanagari spellings a caller (or the STT) may produce.
_ALIASES = {'rahul', 'राहुल', 'rahul sharma', 'राहुल शर्मा', 'sharma', 'शर्मा'}
_FILLERS = re.compile(r'\b(my name is|this is|i am|i\'m|speaking|haan|ji|हाँ|हां|जी|मैं|बोल रहा हूँ|बोल रहा हूं|'
                      r'मेरा नाम|है|hai|main|mera naam|yes|hello|हेलो|हैलो)\b', re.I)


def _norm(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = _FILLERS.sub(' ', text)
    return ' '.join(re.findall(r'[\wऀ-ॿ]+', text))


def name_matches(heard: str) -> bool:
    """True if what the caller said contains the customer's first name (fuzzy, either script)."""
    said = _norm(heard)
    if not said:
        return False
    if said in _ALIASES:
        return True
    tokens = said.split()
    return any(difflib.SequenceMatcher(None, t, alias).ratio() >= 0.8
               for t in tokens for alias in ('rahul', 'राहुल'))


# Right-party check, as in our collection bots ("क्या मेरी बात राहुल शर्मा जी से हो रही है?"): the model classifies
# the caller's answer, the transcript must back it up.
_AFFIRM = {'haan', 'han', 'haa', 'ha', 'haanji', 'ji', 'jee', 'yes', 'yeah', 'yep', 'yup', 'speaking', 'boliye',
           'bolie', 'bolo', 'bataiye', 'batao', 'right', 'correct', 'sahi', 'ok', 'okay',
           'हाँ', 'हां', 'हा', 'जी', 'बोलिए', 'बोलिये', 'बोलो', 'बताइए', 'बताइये', 'बताओ', 'सही', 'ओके', 'यस', 'स्पीकिंग'}
_NEGATE = {'nahi', 'nahin', 'nai', 'no', 'not', 'galat', 'wrong', 'नहीं', 'नही', 'ना', 'गलत', 'ग़लत', 'रॉन्ग'}


def identity_result(answer, name_said, heard):
    """'verified' | 'other_person' | 'unclear' for the caller's reply to "am I speaking with <customer>?".

    answer is the model's reading (yes / no / unclear); heard is the transcript. Verified needs the customer's name
    in the transcript, or a "yes" the transcript backs with an affirmation and no negation.
    """
    if name_said and not name_matches(name_said):
        return 'other_person'
    if name_matches(heard):
        return 'verified'
    if answer == 'no':
        return 'other_person'
    words = set(re.findall(r'[\wऀ-ॿ]+', unicodedata.normalize('NFC', heard).lower()))
    if answer == 'yes' and words & _AFFIRM and not words & _NEGATE:
        return 'verified'
    return 'unclear'
