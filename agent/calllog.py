"""Per-call metadata (disposition, language timeline, voicemail decision, latency) saved as JSON."""
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Per stage (see LatencyObserver), then the assignment's end-of-speech view.
STAGES = ('stt_ms', 'lid_ms', 'endpoint_ms', 'llm_ttft_ms', 'tts_ttfb_ms', 'total_ms',
          'to_llm_first_token_ms', 'to_tts_first_audio_ms')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def percentile(values, p):
    """Nearest-rank percentile; fine for a handful of turns."""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def summarize(turns):
    out = {}
    for stage in STAGES:
        vals = [t[stage] for t in turns if t.get(stage) is not None]
        if vals:
            out[stage] = {'n': len(vals), 'median': round(statistics.median(vals)),
                          'p90': round(percentile(vals, 90))}
    return out


class CallLog:
    def __init__(self, customer_id, out_dir='recordings'):
        self.call_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        self.out_dir = Path(out_dir)
        self.started_clock = None
        self.data = {'call_id': self.call_id, 'customer_id': customer_id, 'started_at': _now(),
                     'ended_at': None, 'disposition': None, 'disposition_details': None,
                     'language_timeline': [], 'language_decisions': [], 'voicemail': None,
                     'turns': [], 'tools': [], 'transcript': [], 'latency_summary': {}, 'recording': None,
                     'audio': None, 'summary': {}}

    def mark_started(self):
        self.started_clock = time.time()
        self.data['started_at'] = _now()

    def add_tool(self, name, arguments, result):
        self.data['tools'].append({'turn': len(self.data['turns']) + 1, 'name': name,
                                   'arguments': arguments, 'result': result, 'at': _now()})

    @staticmethod
    def _message_text(message):
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ' '.join(part.get('text', '') for part in content if isinstance(part, dict))
        return ''

    def set_disposition(self, disposition, **details):
        if self.data['disposition'] is None:  # first outcome wins; later end-of-call paths don't overwrite
            self.data['disposition'], self.data['disposition_details'] = disposition, details or None

    def save(self, language_tracker=None, messages=None):
        self.data['ended_at'] = _now()
        if self.data['disposition'] is None:
            self.data['disposition'] = 'disconnected'
        if language_tracker:
            self.data['language_timeline'] = language_tracker.timeline
            self.data['language_decisions'] = language_tracker.decisions
        if messages:
            self.data['transcript'] = [{'role': message['role'], 'text': self._message_text(message)}
                                       for message in messages
                                       if isinstance(message, dict)
                                       and message.get('role') in ('user', 'assistant')
                                       and self._message_text(message).strip()]
        self.data['latency_summary'] = summarize(self.data['turns'])
        started = datetime.fromisoformat(self.data['started_at'])
        ended = datetime.fromisoformat(self.data['ended_at'])
        self.data['summary'] = {'duration_s': round((ended - started).total_seconds(), 2),
                                'turn_count': len(self.data['turns']),
                                'tool_call_count': len(self.data['tools']),
                                'language_switch_count': max(0, len(self.data['language_timeline']) - 1)}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f'{self.call_id}.json'
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str))
        return path
