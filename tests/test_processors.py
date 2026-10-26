import asyncio
import time
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.azure.tts import AzureTTSService
from pipecat.tests.utils import SleepFrame, run_test

import agent.processors as procs
from agent.language import LanguageTracker
from agent.lid import LanguageDetector
from pipecat.frames.frames import InterimTranscriptionFrame
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from agent.processors import AsrFeed, BilingualTranscriber, LatencyObserver, VoicemailGate
from agent.voicemail import VoicemailDetector


def tf(text):
    return TranscriptionFrame(text=text, user_id='u', timestamp='')


def tfl(text, lang):
    return TranscriptionFrame(text=text, user_id='u', timestamp='', language=lang)


async def test_bilingual_merge_switches_before_the_turn():
    async def on_switch(lang, proc):
        await proc.push_frame(TTSUpdateSettingsFrame(delta=AzureTTSService.Settings(voice=lang)))

    bi = BilingualTranscriber(LanguageDetector(client=None), LanguageTracker('hi'), on_switch, lambda: [])
    vad_start, vad_stop = VADUserStartedSpeakingFrame(start_secs=0.2), VADUserStoppedSpeakingFrame(stop_secs=0.2)

    down, _ = await run_test(bi, frames_to_send=[
        vad_start, tfl('I will pay the full amount on Friday', 'en-IN'), vad_stop,
        tfl('I will pay the full amount on Friday', 'hi-IN'), SleepFrame(0.8),
        tfl('late duplicate final', 'en-IN'),  # after the merge: dropped
    ], expected_down_frames=[VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame, TTSUpdateSettingsFrame,
                             RTVIServerMessageFrame, TranscriptionFrame])
    merged, event = down[-1], down[-2].data
    # clear-cut English (Latin text even from the Hindi recogniser) on an 8-word turn: immediate switch,
    # TTS settings land before the transcript, and the transcript is the en-IN recogniser's
    assert bi.tracker.lang == 'en' and str(merged.language) == 'en-IN'
    assert merged.result['total_results'] == {'hi-IN': 'I will pay the full amount on Friday',
                                              'en-IN': 'I will pay the full amount on Friday'}
    # the console's turn card: both transcripts, the resolved language and the text sent to the LLM
    assert event['kind'] == 'turn' and event['data']['resolved'] == 'en' and event['data']['switched']
    assert event['data']['text'] == merged.text


async def test_asr_feed_streams_both_languages():
    frames = [InterimTranscriptionFrame(text='मेरा नाम', user_id='u', timestamp='', language='hi-IN'),
              tfl('Mera naam Rahul hai.', 'en-IN')]
    down, _ = await run_test(AsrFeed(), frames_to_send=frames, expected_down_frames=[
        RTVIServerMessageFrame, InterimTranscriptionFrame, RTVIServerMessageFrame, TranscriptionFrame])
    assert [d.data['data'] for d in down if isinstance(d, RTVIServerMessageFrame)] == [
        {'lang': 'hi-IN', 'text': 'मेरा नाम', 'final': False}, {'lang': 'en-IN', 'text': 'Mera naam Rahul hai.', 'final': True}]


async def test_gate_holds_user_frames_until_classifier_decides():
    texts = []

    def classify(t):
        texts.append(t)
        return {'voicemail': len(t) == 2, 'score': 0.9, 'source': 'classifier'}

    gate = VoicemailGate(VoicemailDetector(), None, None, classify=classify)
    gate.t0 = time.time()  # deciding, watcher not started
    await run_test(gate, frames_to_send=[tfl('please leave a message after the beep', 'en-IN'),
                                         tfl('कृपया बीप के बाद संदेश छोड़ें', 'hi-IN'), SleepFrame(0.1)],
                   expected_down_frames=[])  # user frames are held while deciding
    assert gate.detector.decision['kind'] == 'voicemail' and gate.detector.decision['reason'] == 'classifier score 0.9'
    assert texts[-1] == ['please leave a message after the beep', 'कृपया बीप के बाद संदेश छोड़ें']  # both recognisers

    gate.open = True  # after a 'human' decision everything flows
    await run_test(gate, frames_to_send=[tf('hello')], expected_down_frames=[TranscriptionFrame])


async def test_gate_leaves_voicemail_once_quiet():
    calls = []
    det = VoicemailDetector()
    gate = VoicemailGate(det, None, lambda d: calls.append(d) or asyncio.sleep(0), classify=None)
    gate.push_frame = lambda *a, **k: asyncio.sleep(0)  # not in a pipeline here
    gate.t0 = time.time()
    det.classified({'voicemail': True, 'score': 0.9, 'source': 'classifier'}, 0.5)  # no VAD segments at all
    start = time.time()
    await asyncio.wait_for(gate._watch(), 3)
    assert calls and calls[0]['kind'] == 'voicemail' and time.time() - start < 1.5


async def test_latency_observer_stages(monkeypatch):
    rows, clock = [], [100.0]
    monkeypatch.setattr(procs.time, 'time', lambda: clock[0])
    obs = LatencyObserver(lambda r: rows.append(r) or asyncio.sleep(0))

    async def push(frame, at):
        clock[0] = at
        await obs.on_push_frame(SimpleNamespace(frame=frame))

    await push(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=99.0), 99.0)
    await push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=100.2), 100.2)  # speech ended at 100.0
    await push(tf('haan ji'), 100.35)  # raw recogniser final
    merged = TranscriptionFrame(text='haan ji', user_id='u', timestamp='', result={'merged': True})
    await push(merged, 100.9)          # after merge + language ID
    await push(UserStoppedSpeakingFrame(), 101.0)
    await push(MetricsFrame(data=[TTFBMetricsData(processor='OpenAILLMService#0', value=0.42)]), 101.4)
    await push(LLMTextFrame('ok'), 101.42)
    await push(MetricsFrame(data=[TTFBMetricsData(processor='AzureTTSService#0', value=0.18)]), 101.6)
    await push(MetricsFrame(data=[TTFBMetricsData(processor='OpenAILLMService#0', value=0.9)]), 101.61)  # 2nd call: ignored
    await push(TTSAudioRawFrame(audio=b'\0\0', sample_rate=24000, num_channels=1), 101.62)
    await push(BotStartedSpeakingFrame(), 101.7)
    r = rows[0]
    # each stage on its own; LLM/TTS come from Pipecat's TTFB metrics (time to first token / first audio)
    assert (r['stt_ms'], r['lid_ms'], r['endpoint_ms'], r['llm_ttft_ms'], r['tts_ttfb_ms'], r['total_ms']) == \
        (350, 550, 1000, 420, 180, 1700)
    assert (r['to_llm_first_token_ms'], r['to_tts_first_audio_ms']) == (1420, 1620)


async def test_latency_observer_flushes_ended_turn_without_reply(monkeypatch):
    rows, clock = [], [200.0]
    monkeypatch.setattr(procs.time, 'time', lambda: clock[0])
    observer = LatencyObserver(lambda row: rows.append(row) or asyncio.sleep(0))

    async def push(frame, at):
        clock[0] = at
        await observer.on_push_frame(SimpleNamespace(frame=frame))

    await push(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=199.8), 199.8)
    await push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=201.0), 201.0)
    await push(tf('hello'), 201.3)
    merged = TranscriptionFrame(text='hello', user_id='u', timestamp='', result={'merged': True})
    await push(merged, 201.4)
    await push(UserStoppedSpeakingFrame(), 202.0)
    await observer.flush()

    assert len(rows) == 1
    assert rows[0]['response_status'] == 'no_response'
    assert rows[0]['stt_ms'] == 500 and rows[0]['lid_ms'] == 100 and rows[0]['endpoint_ms'] == 1200
    assert rows[0]['llm_ttft_ms'] is None and rows[0]['tts_ttfb_ms'] is None and rows[0]['total_ms'] is None


async def test_gate_waits_for_both_recognisers():
    scores = {'द पर्सन यू आर ट्राइंग टू रीच': 0.04, 'the person you are trying to reach': 1.0}

    def classify(texts):
        score = max(scores.get(t, 0.0) for t in texts)
        return {'voicemail': score >= 0.45, 'score': score, 'source': 'classifier'}

    gate = VoicemailGate(VoicemailDetector(), None, None, classify=classify)
    gate.t0 = time.time()
    await run_test(gate, frames_to_send=[VADUserStartedSpeakingFrame(start_secs=0.2),
                                         tfl('द पर्सन यू आर ट्राइंग टू रीच', 'hi-IN'), SleepFrame(0.05)],
                   expected_down_frames=[])
    assert gate.detector.hold_until > 0 and gate.detector.decision is None  # waiting for en-IN
    await run_test(gate, frames_to_send=[tfl('the person you are trying to reach', 'en-IN'), SleepFrame(0.05)],
                   expected_down_frames=[])
    assert gate.detector.hold_until == 0.0 and gate.detector.decision['kind'] == 'voicemail'
