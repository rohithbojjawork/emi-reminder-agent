"""Custom Pipecat pieces: live ASR feed, voicemail gate, bilingual merge + language ID, latency observer.

Live call events go to the browser console (web/call.html) as RTVI server messages: {'kind': ..., 'data': ...}.
"""
import asyncio
import json
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    FunctionCallInProgressFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.utils.time import time_now_iso8601

from agent.lid import STT_CODES
from agent.voicemail import FINAL_GRACE_S, is_tone

# Frames held back from the LLM side until we know a person picked up.
_USER_FRAMES = (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame, UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame, TranscriptionFrame, InterimTranscriptionFrame)


def server_message(kind, data):
    """RTVI server message for the console; JSON round-trip so enums, dates etc. serialise."""
    return RTVIServerMessageFrame(data={'kind': kind, 'data': json.loads(json.dumps(data, default=str, ensure_ascii=False))})


class AsrFeed(FrameProcessor):
    """Streams every partial and final transcript from each recogniser to the console, tagged with language."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)) and frame.text.strip():
            await self.push_frame(server_message('asr', {'lang': str(frame.language), 'text': frame.text,
                                                         'final': isinstance(frame, TranscriptionFrame)}))
        await self.push_frame(frame, direction)


class VoicemailGate(FrameProcessor):
    """Runs VoicemailDetector on the first seconds of the call and gates the conversation on it.

    Sits after STT, so it sees VAD frames, transcripts and the raw audio (STT passes audio through).
    Until a decision, user turn frames stop here, so the LLM never answers a voicemail greeting.
    Audio always passes, so the recording keeps the callee's side.
    """

    def __init__(self, detector, on_human, on_voicemail, classify, recognisers=2, **kwargs):
        """classify(texts) -> {'voicemail': bool, 'score', ...}: agent.vm_classifier.VoicemailClassifier.classify."""
        super().__init__(**kwargs)
        self.detector, self.on_human, self.on_voicemail, self.classify = detector, on_human, on_voicemail, classify
        self.t0 = None
        self.open = False
        self.finals = {}  # language -> final transcripts so far
        self.recognisers = recognisers
        self._utterance_langs = set()  # recognisers that sent a final for the current utterance
        self._audio = bytearray()
        self._tone_windows = 0

    def start(self):
        self.t0 = time.time()
        self.create_task(self._watch())

    async def _watch(self):
        while self.detector.tick(time.time() - self.t0) is None:
            await asyncio.sleep(0.1)
        decision = self.detector.decision
        logger.info(f'VOICEMAIL_DECISION {decision}')
        await self.push_frame(server_message('voicemail', decision))
        if decision['kind'] == 'human':
            self.open = True
            await self.on_human(decision)
            return
        # Voicemail: wait for the greeting to finish (beep, or 1 s of quiet; at most 12 s) before speaking.
        waited = time.time()
        while time.time() - waited < 12:
            last = self.detector.segments[-1] if self.detector.segments else None
            speaking = last is not None and last[1] is None
            quiet_for = 0.0 if speaking else time.time() - self.t0 - (last[1] if last else 0.0)
            if (self.detector.beep_at or 0) >= decision['decided_at_s'] or quiet_for >= 1.0:
                break
            await asyncio.sleep(0.1)
        await self.on_voicemail(decision)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self.t0 is not None and not self.open:  # still deciding, or waiting out a voicemail greeting
            self._observe(frame)
            if isinstance(frame, _USER_FRAMES):
                return
        await self.push_frame(frame, direction)

    def _observe(self, frame):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._utterance_langs = set()
            self.detector.speech_start(frame.timestamp - frame.start_secs - self.t0)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.detector.speech_stop(frame.timestamp - frame.stop_secs - self.t0)
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip() and self.detector.decision is None:
            self.detector.transcript_text(frame.text, time.time() - self.t0)
            self.finals.setdefault(str(frame.language), []).append(frame.text.strip())
            # The hi-IN recogniser writes English announcements in Devanagari, which scores low; the en-IN final
            # may land later, so a 'not voicemail' waits for every recogniser (or FINAL_GRACE_S).
            self._utterance_langs.add(str(frame.language))
            now = time.time() - self.t0
            self.detector.hold_until = 0.0 if len(self._utterance_langs) >= self.recognisers else now + FINAL_GRACE_S
            result = self.classify([' '.join(parts) for parts in self.finals.values()])  # ~0.03 ms: inline
            logger.info(f'VOICEMAIL_CLASSIFY {result}')
            self.detector.classified(result, now)
        elif isinstance(frame, InputAudioRawFrame):
            self._audio += frame.audio
            window = frame.sample_rate // 10 * 2  # 100 ms of 16-bit mono
            while len(self._audio) >= window:
                chunk, self._audio = bytes(self._audio[:window]), self._audio[window:]
                self._tone_windows = self._tone_windows + 1 if is_tone(chunk, frame.sample_rate) else 0
                if self._tone_windows == 2:  # >= 200 ms of steady tone
                    self.detector.beep(time.time() - self.t0)


class BilingualTranscriber(FrameProcessor):
    """Merges the hi-IN and en-IN recognizers' finals for one utterance into total_results, resolves
    the turn's language, reconfigures TTS and prompt on a switch, then forwards one
    TranscriptionFrame: the resolved language's transcript. Slower recognisers get GRACE_S after the first.

    The merge runs as a task so audio and VAD frames keep flowing while language ID waits on the LLM.
    """

    GRACE_S = 0.6

    def __init__(self, detector, tracker, on_switch, history, **kwargs):
        super().__init__(**kwargs)
        self.detector, self.tracker, self.on_switch, self.history = detector, tracker, on_switch, history
        self.buf = self._empty()
        self.speaking = False
        self.emitted = False  # emitted since the last VAD start: late finals from a slow recogniser are dropped
        self.turn = 0
        self._task = None

    @staticmethod
    def _empty():
        return {code: [] for code in STT_CODES.values()}

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            code = str(frame.language)
            if not self.emitted and code in self.buf and frame.text.strip():
                self.buf[code].append(frame.text.strip())
                self._schedule()
            return
        if isinstance(frame, InterimTranscriptionFrame):  # only the active language reaches the turn logic
            if str(frame.language) == STT_CODES[self.tracker.lang]:
                await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.speaking, self.emitted = True, False
            if self._task and not self._task.done():
                await self.cancel_task(self._task)  # caller kept talking: keep buffering this utterance
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.speaking = False
            self._schedule()
        await self.push_frame(frame, direction)

    def _schedule(self):
        if not self.speaking and any(self.buf.values()) and not (self._task and not self._task.done()):
            self._task = self.create_task(self._emit())

    async def _emit(self):
        start = time.monotonic()
        while not all(self.buf.values()) and time.monotonic() - start < self.GRACE_S:
            await asyncio.sleep(0.05)
        total = {code: ' '.join(parts) for code, parts in self.buf.items()}
        detection = await self.detector.detect(total, self.tracker.lang, self.history())
        self.turn += 1
        switched = self.tracker.observe(self.turn, detection, max(len(t.split()) for t in total.values()))
        decision = self.tracker.decisions[-1]
        if switched:
            await self.on_switch(switched, self)
        lang = self.tracker.lang
        text = total[STT_CODES[lang]] or next(t for t in total.values() if t)
        logger.info(f'TURN {self.turn} resolved={lang} {decision["decision"]}: {decision["reason"]}')
        await self.push_frame(server_message('turn', {**decision, 'text': text, 'switched': bool(switched)}))
        self.buf, self.emitted = self._empty(), True
        await self.push_frame(TranscriptionFrame(text, 'caller', time_now_iso8601(), STT_CODES[lang],
                                                 result={'merged': True, 'total_results': total}, finalized=True))


class ReplyTracker(FrameProcessor):
    """Sits between LLM and TTS and remembers the full reply the agent meant to say (LLM text or a scripted
    TTSSpeakFrame line), so a barge-in can be compared with what was actually spoken."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intended = ''

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.intended = ''
        elif isinstance(frame, LLMTextFrame):
            self.intended += frame.text
        elif isinstance(frame, TTSSpeakFrame):
            self.intended = frame.text
        await self.push_frame(frame, direction)


class LatencyObserver(BaseObserver):
    """One row per turn, each stage on its own (no cumulative "complete" times):
    stt_ms        estimated end of speech (VAD stop - stop_secs) -> last recogniser final
    lid_ms        recogniser final -> resolved transcript (merge + language ID)
    endpoint_ms   end of speech -> turn-end decision (turn strategy)
    llm_ttft_ms   LLM request -> first token (Pipecat TTFB metric of the LLM service)
    tts_ttfb_ms   TTS request -> first audio (Pipecat TTFB metric of the TTS service)
    total_ms      end of speech -> bot starts speaking
    """

    def __init__(self, on_turn, **kwargs):
        super().__init__(**kwargs)
        self.on_turn = on_turn
        self.turn = None
        self.count = 0
        self._seen = set()

    async def on_push_frame(self, data: FramePushed):
        frame, now = data.frame, time.time()
        if frame.id in self._seen:  # a frame is pushed once per hop; time its first appearance only
            return
        self._seen.add(frame.id)
        t = self.turn
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if t and 'turn_end' in t and 'bot_started' not in t:
                await self._finish(t, response_status='no_response')
                self.turn = t = None
            if t and 'bot_started' not in t:
                t['resumed'] = True  # user kept talking; the next VAD stop starts the turn again
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not t or 'bot_started' in t or t.get('resumed'):
                self.turn = t = {}
            t.update(speech_end=frame.timestamp - frame.stop_secs, vad_stop_secs=frame.stop_secs)
            t.pop('resumed', None)
        elif not t or 'bot_started' in t:
            return
        elif isinstance(frame, TranscriptionFrame):
            if isinstance(frame.result, dict) and frame.result.get('merged'):  # after merge + language ID
                t['transcript_ready'] = now
                t['text'] = frame.text
            else:
                t['stt_final'] = now  # latest raw recogniser final
        elif isinstance(frame, UserStoppedSpeakingFrame):
            t.setdefault('turn_end', now)
        elif isinstance(frame, FunctionCallInProgressFrame):
            t.setdefault('tools', []).append(frame.function_name)
        elif isinstance(frame, MetricsFrame) and 'turn_end' in t:
            for m in frame.data:
                if isinstance(m, TTFBMetricsData) and m.value:
                    key = 'llm_ttft' if 'LLM' in m.processor else 'tts_ttfb' if 'TTS' in m.processor else None
                    if key:
                        t.setdefault(key, m.value * 1000)  # first call of the turn
        elif isinstance(frame, (LLMTextFrame, TTSAudioRawFrame)) and 'turn_end' in t:
            t.setdefault('llm_first_token' if isinstance(frame, LLMTextFrame) else 'tts_first_audio', now)
        elif isinstance(frame, BotStartedSpeakingFrame) and 'turn_end' in t:
            if 'bot_started' not in t:
                t['bot_started'] = now
                await self._finish(t)

    async def flush(self):
        """Persist an ended user turn even when the agent never started a reply."""
        t = self.turn
        if t and 'turn_end' in t and 'bot_started' not in t:
            await self._finish(t, response_status='no_response')

    async def _finish(self, t, response_status='spoken'):
        self.count += 1
        ms = lambda a, b='speech_end': round((t[a] - t[b]) * 1000) if a in t and b in t else None
        row = {'turn': self.count, 'text': t.get('text'), 'tools': t.get('tools', []), 'vad_stop_secs': t['vad_stop_secs'],
               'speech_end_epoch': round(t['speech_end'], 3),
               'response_status': response_status,
               'stt_ms': ms('stt_final'), 'lid_ms': ms('transcript_ready', 'stt_final'), 'endpoint_ms': ms('turn_end'),
               'llm_ttft_ms': round(t['llm_ttft']) if 'llm_ttft' in t else None,
               'tts_ttfb_ms': round(t['tts_ttfb']) if 'tts_ttfb' in t else None,
               'total_ms': ms('bot_started'),
               # assignment view: from end of speech to first LLM token / first TTS audio
               'to_llm_first_token_ms': ms('llm_first_token'), 'to_tts_first_audio_ms': ms('tts_first_audio')}
        logger.info(f'TURN_LATENCY {row}')
        self._seen.clear()
        if self.turn is t:
            self.turn = None
        await self.on_turn(row)
