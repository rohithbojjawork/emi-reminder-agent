"""Answering-machine detection: transcript phrases identify voicemail; timing rules decide human pickup."""
import numpy as np

HUMAN_MAX_GREETING = 2.5   # s: a person's opening "hello?" / "haan ji" is short
HUMAN_PAUSE = 0.8          # s: ...and then they wait for us
GAP_MERGE = 0.7            # s: pauses shorter than this are one stretch of speech
SILENT_PICKUP = 3.0        # s: no speech at all -> a person waiting for us (greetings start within ~1s)
DECIDE_BY = 10.0           # s: undecided by now -> treat as human
FINAL_GRACE_S = 1.0        # s: after one recogniser's final, wait this long for the other's before a 'not voicemail'


def is_tone(samples, sample_rate):
    """True if a 16-bit PCM chunk is dominated by one 400-2000 Hz frequency (a beep)."""
    x = np.frombuffer(samples, dtype=np.int16).astype(np.float32) if isinstance(samples, (bytes, bytearray)) \
        else np.asarray(samples, dtype=np.float32)
    if len(x) < sample_rate // 50 or np.sqrt(np.mean(x ** 2)) < 300:  # need >=20 ms and audible level
        return False
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    peak = int(np.argmax(spec))
    band = spec[max(peak - 2, 0):peak + 3].sum()
    return 400 <= freqs[peak] <= 2000 and band / spec.sum() > 0.6


class VoicemailDetector:
    """Feed events with times in seconds since the call connected; read .decision once set."""

    def __init__(self):
        self.segments = []      # [start, end or None] of callee speech, short gaps merged
        self.transcript = []
        self.classifications = []
        self.beep_at = None
        self.hold_until = 0.0  # 'not voicemail' can't decide human before this (other recogniser pending)
        self.decision = None    # {'kind': 'human'|'voicemail', 'reason': ..., 'signals': ...}

    def speech_start(self, t):
        if self.segments and self.segments[-1][1] is not None and t - self.segments[-1][1] < GAP_MERGE:
            self.segments[-1][1] = None  # resume the same segment
        else:
            self.segments.append([t, None])

    def speech_stop(self, t):
        if self.segments and self.segments[-1][1] is None:
            self.segments[-1][1] = t

    def transcript_text(self, text, t):
        self.transcript.append((round(t, 2), text))

    def classified(self, result, t):
        """Apply a classification of the transcript so far."""
        self.classifications.append({**result, 'at_s': round(t, 2)})
        if self.decision is None and result['voicemail']:
            reason = result.get('reason') or f'{result.get("source", "classifier")} score {result.get("score", 0)}'
            self._decide('voicemail', reason, t)

    def beep(self, t):
        if self.beep_at is None:
            self.beep_at = t

    def tick(self, t):
        """Call regularly; applies the human-side timing rules."""
        if self.decision is not None:
            return self.decision
        if self.segments:
            first, last = self.segments[0], self.segments[-1]
            paused = last[1] is not None and t - last[1] >= HUMAN_PAUSE
            if paused and len(self.segments) == 1 and first[1] - first[0] <= HUMAN_MAX_GREETING:
                self._decide('human', 'short greeting followed by a pause', t)
            elif paused and self.classifications and not self.classifications[-1]['voicemail'] and t >= self.hold_until:
                self._decide('human', 'classifier said not voicemail and the callee paused', t)
        elif t >= SILENT_PICKUP:
            self._decide('human', f'silent pickup: no speech in {SILENT_PICKUP}s', t)
        if self.decision is None and t >= DECIDE_BY:
            self._decide('human', f'no voicemail evidence by {DECIDE_BY}s (fail-safe)', t)
        return self.decision

    def _decide(self, kind, reason, t):
        longest = max(((e if e is not None else t) - s for s, e in self.segments), default=0.0)
        self.decision = {'kind': kind, 'reason': reason, 'decided_at_s': round(t, 2), 'signals': {
            'segments': [[round(s, 2), None if e is None else round(e, 2)] for s, e in self.segments],
            'longest_speech_s': round(longest, 2), 'beep_at_s': self.beep_at,
            'transcript': self.transcript, 'classifications': self.classifications}}
