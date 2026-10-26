"""Deterministic wording of amounts and dates for TTS, so the LLM never writes a number itself.

Uses the Indian system (thousand, lakh, crore) in both languages, the way callers say it:
4500 -> "four thousand five hundred rupees" / "चार हज़ार पाँच सौ रुपये".
"""
from datetime import date

from agent.hindi_numbers import HI_0_99

EN_0_19 = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
           'fifteen sixteen seventeen eighteen nineteen').split()
EN_TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
EN_ORDINAL = {'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', 'eight': 'eighth',
              'nine': 'ninth', 'twelve': 'twelfth'}
EN_MONTHS = ('January February March April May June July August September October November '
             'December').split()
HI_MONTHS = 'जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर'.split()
EN_DAYS = 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
HI_DAYS = 'सोमवार मंगलवार बुधवार गुरुवार शुक्रवार शनिवार रविवार'.split()

# (value, english word, hindi word), largest first
UNITS = ((10_000_000, 'crore', 'करोड़'), (100_000, 'lakh', 'लाख'), (1000, 'thousand', 'हज़ार'),
         (100, 'hundred', 'सौ'))


def _below_100(n, lang):
    if lang == 'hi':
        return HI_0_99[n]
    if n < 20:
        return EN_0_19[n]
    return EN_TENS[n // 10] + ('' if n % 10 == 0 else ' ' + EN_0_19[n % 10])


def number_words(n, lang):
    """Cardinal number in words, Indian grouping. lang is 'en' or 'hi'."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f'expected a non-negative int, got {n!r}')
    if n < 100:
        return _below_100(n, lang)
    parts = []
    for value, en, hi in UNITS:
        if n >= value:
            parts.append(f'{number_words(n // value, lang)} {hi if lang == "hi" else en}')
            n %= value
    if n:
        parts.append(_below_100(n, lang))
    return ' '.join(parts)


def amount_words(rupees, lang):
    """Whole-rupee amount as spoken: 4500 -> 'four thousand five hundred rupees'."""
    if lang == 'hi':
        return f'{number_words(rupees, "hi")} रुपये'
    return f'{number_words(rupees, "en")} rupee{"" if rupees == 1 else "s"}'


def _en_ordinal(n):
    words = number_words(n, 'en').split()
    last = words[-1]
    if last in EN_ORDINAL:
        words[-1] = EN_ORDINAL[last]
    elif last.endswith('y'):
        words[-1] = last[:-1] + 'ieth'
    else:
        words[-1] = last + 'th'
    return ' '.join(words)


def date_words(d: date, lang, weekday=False):
    """'the tenth of October' / 'दस अक्टूबर'; weekday=True prefixes the day name for read-backs."""
    if lang == 'hi':
        text = f'{number_words(d.day, "hi")} {HI_MONTHS[d.month - 1]}'
        return f'{HI_DAYS[d.weekday()]}, {text}' if weekday else text
    text = f'the {_en_ordinal(d.day)} of {EN_MONTHS[d.month - 1]}'
    return f'{EN_DAYS[d.weekday()]}, {text}' if weekday else text
