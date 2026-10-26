"""EMI collections reminder voice agent (Pipecat 1.14, Azure speech, browser WebRTC).

Run: uv run bot.py -t webrtc   then open http://localhost:7860/call
"""
import asyncio
import json
import os
import random
import re
import wave
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    EndWorkerFrame,
    FunctionCallResultProperties,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIFunctionCallReportLevel, RTVIObserverParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from agent.calllog import CallLog
from agent.language import LanguageTracker
from agent.lid import LANG_PROMPT, STT_CODES, LanguageDetector
from agent.processors import AsrFeed, BilingualTranscriber, LatencyObserver, ReplyTracker, VoicemailGate, server_message
from agent.flow import CallFlow
from agent.prompts import (DISPOSITIONS, GREETING, VOICEMAIL_MESSAGES, interruption_note, language_switch_note,
                           system_prompt)
from agent.record import CUSTOMER
from agent.vm_classifier import VoicemailClassifier
from agent.voicemail import VoicemailDetector

load_dotenv(override=True)

LANG_CODES = {'hi': Language.HI_IN, 'en': Language.EN_IN}
VOICES = {'hi': 'hi-IN-SwaraNeural', 'en': 'en-IN-NeerjaNeural'}
VAD_STOP_SECS = 0.2  # raw VAD silence before "stopped speaking"; the turn-end wait is set by TURN_MODE

TURN_MODE = os.getenv('TURN_MODE', 'fast')
LANG_SWITCH_MODE = os.getenv('LANG_SWITCH_MODE', 'implicit')  # implicit | explicit | custom
LID_PROMPT_FILE = os.getenv('LID_PROMPT_FILE')  # custom mode: your own language-switch prompt
MIN_WORDS_FOR_LANG_SWITCH = int(os.getenv('MIN_WORDS_FOR_LANG_SWITCH', '2'))  # minWordsForLangSwitch
INITIAL_MESSAGE_BARGE = os.getenv('INITIAL_MESSAGE_BARGE', 'true') == 'true'  # initial_message_barge (ours: true)

TOOLS = ToolsSchema(standard_tools=[
    FunctionSchema(name='verify_identity',
                   description='Record the caller\'s answer to "am I speaking with the customer?". Call on every answer '
                               'about who they are.',
                   properties={'answer': {'type': 'string', 'enum': ['yes', 'no', 'unclear'],
                                          'description': 'yes = they are the customer; no = someone else.'},
                               'name_said': {'type': 'string', 'description': 'A name the caller gave for themselves, '
                                             'exactly as said. Omit if they gave none.'}},
                   required=['answer']),
    FunctionSchema(name='repeat_reminder', description='Speak the EMI amount and due date again, when the verified '
                   'customer asks for them.', properties={}, required=[]),
    FunctionSchema(name='record_promise_to_pay',
                   description='Record the date by which the verified customer promises to pay the overdue EMI.',
                   properties={'date': {'type': 'string', 'description': 'Promised date as YYYY-MM-DD, from the calendar.'}},
                   required=['date']),
    FunctionSchema(name='end_call', description='Record the call outcome, speak the closing line and hang up.',
                   properties={'disposition': {'type': 'string', 'enum': DISPOSITIONS},
                               'callback_time': {'type': 'string', 'description': 'For callback_requested: when to '
                                                 'call, as the caller said it (e.g. "tomorrow 5 pm").'}},
                   required=['disposition']),
])


def _text(message):
    content = message.get('content') if isinstance(message, dict) else None
    if isinstance(content, list):
        return ' '.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ''


def _last(context, role):
    return next((_text(m) for m in reversed(context.messages) if isinstance(m, dict) and m.get('role') == role), '')


async def run_bot(transport, runner_args: RunnerArguments):
    today = date.fromisoformat(os.getenv('TODAY')) if os.getenv('TODAY') else date.today()
    log = CallLog(CUSTOMER['customer_id'])
    flow = CallFlow(CUSTOMER['preferred_language'], today)  # tool rules (agent/flow.py)
    introduced = {flow.lang}  # languages the AI + recording disclosure has been spoken in

    # --- LLM and language ID ---
    llm_key = (os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or '').strip().removeprefix('Bearer ').strip()
    llm_url, llm_model = os.getenv('LLM_BASE_URL') or None, os.getenv('LLM_MODEL', 'gpt-4.1-mini')
    llm = OpenAILLMService(api_key=llm_key, base_url=llm_url, settings=OpenAILLMService.Settings(
        model=llm_model, temperature=0.2, system_instruction=system_prompt(flow.lang, today)))
    lid_prompt = open(LID_PROMPT_FILE).read() if LANG_SWITCH_MODE == 'custom' and LID_PROMPT_FILE else LANG_PROMPT
    lid_mode = 'explicit' if LANG_SWITCH_MODE == 'explicit' or (
        LANG_SWITCH_MODE == 'custom' and lid_prompt.lstrip().lower().startswith('<explicit_language_prompt>')) else 'implicit'
    side_llm = AsyncOpenAI(api_key=llm_key, base_url=llm_url)  # language ID
    side_model = os.getenv('LID_MODEL') or llm_model
    detector = LanguageDetector(client=side_llm, model=side_model, mode=lid_mode, prompt=lid_prompt,
                                min_words=MIN_WORDS_FOR_LANG_SWITCH)
    tracker = LanguageTracker(flow.lang, mode=lid_mode)

    # --- Azure speech: one streaming recogniser per language in parallel ---
    azure = {'api_key': os.environ['AZURE_SPEECH_KEY'], 'region': os.environ['AZURE_SPEECH_REGION']}
    tts_settings = lambda lang: AzureTTSService.Settings(voice=VOICES[lang], language=LANG_CODES[lang])
    tts = AzureTTSService(**azure, settings=tts_settings(flow.lang))
    stt = ParallelPipeline(*[[AzureSTTService(**azure, settings=AzureSTTService.Settings(
        language=LANG_CODES[lang], segmentation_silence_timeout_ms=500))] for lang in STT_CODES])

    # --- tools: every number the caller hears is spoken by these (TTSSpeakFrame), never written by the model ---
    async def speak_and_wait(params, result, text):
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
        await params.llm.push_frame(TTSSpeakFrame(text))  # appended to context, so the model knows it was said

    async def run_tool(params, name, arguments, result, say):
        log.add_tool(name, arguments, result)
        if say:
            await speak_and_wait(params, result, say)
        else:
            await params.result_callback(result)

    async def verify_identity(params: FunctionCallParams):
        result, say = flow.verify_identity(params.arguments, _last(params.context, 'user'))
        await run_tool(params, 'verify_identity', params.arguments, result, say)

    async def repeat_reminder(params: FunctionCallParams):
        result, say = flow.repeat_reminder(params.arguments)
        await run_tool(params, 'repeat_reminder', params.arguments, result, say)

    async def record_promise_to_pay(params: FunctionCallParams):
        result, say = flow.record_promise_to_pay(params.arguments)
        await run_tool(params, 'record_promise_to_pay', params.arguments, result, say)

    async def end_call(params: FunctionCallParams):
        result, goodbye, details = flow.end_call(params.arguments)
        log.add_tool('end_call', params.arguments, result)
        if not goodbye:
            return await params.result_callback(result)
        disposition = params.arguments['disposition']
        log.set_disposition(disposition, **details)
        await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
        await params.llm.push_frame(server_message('disposition', {'disposition': disposition, **details}))
        await params.llm.push_frame(TTSSpeakFrame(goodbye))
        await params.llm.push_frame(EndWorkerFrame(reason=disposition))

    llm.register_function('verify_identity', verify_identity)
    llm.register_function('repeat_reminder', repeat_reminder)
    llm.register_function('record_promise_to_pay', record_promise_to_pay)
    llm.register_function('end_call', end_call, cancel_on_interruption=False)

    # --- turns: default start strategies interrupt the bot as soon as VAD hears the caller (~0.2 s) ---
    stop = {'prod': SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.5),
            'fast': SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6),
            'smart': TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())}[TURN_MODE]
    context = LLMContext(tools=TOOLS)
    user_agg, assistant_agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(stop=[stop]),
        user_mute_strategies=[] if INITIAL_MESSAGE_BARGE else [MuteUntilFirstBotCompleteUserMuteStrategy()]))

    # --- language: on a switch, TTS voice and prompt change before the LLM sees the turn ---
    async def on_switch(new_lang, proc):
        flow.lang = new_lang
        if new_lang not in introduced:  # the caller may not have understood the disclosure in the other language
            introduced.add(new_lang)
            context.add_message({'role': 'system', 'content': language_switch_note(new_lang)})
        await proc.push_frame(TTSUpdateSettingsFrame(delta=tts_settings(new_lang)))
        await proc.push_frame(LLMUpdateSettingsFrame(delta=OpenAILLMService.Settings(
            system_instruction=system_prompt(new_lang, today))))

    def lid_history():
        """(role, lang, text) turns for the language-ID prompt; user turns carry their resolved language."""
        langs = iter(d['resolved'] for d in tracker.decisions)
        return [(m['role'], next(langs, None) if m['role'] == 'user' else None, _text(m))
                for m in context.messages if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and _text(m)]

    # --- voicemail: greet only once a person is detected; otherwise one random scripted message, then hang up ---
    async def on_human(decision):
        log.data['voicemail'] = decision
        await worker.queue_frames([TTSSpeakFrame(GREETING[flow.lang])])

    async def on_voicemail(decision):
        log.data['voicemail'] = decision
        log.set_disposition('voicemail', reason=decision['reason'])
        message = random.choice(VOICEMAIL_MESSAGES[flow.lang])
        await worker.queue_frames([server_message('disposition', {'disposition': 'voicemail', 'message': message}),
                                   TTSSpeakFrame(message, append_to_context=False), EndFrame()])

    async def on_turn(row):
        if log.started_clock is not None:
            row['audio_offset_s'] = round(max(0, row['speech_end_epoch'] - log.started_clock), 3)
        log.data['turns'].append(row)
        if call_running:
            await worker.queue_frames([server_message('latency', row)])

    gate = VoicemailGate(VoicemailDetector(), on_human, on_voicemail, classify=VoicemailClassifier().classify,
                         recognisers=len(STT_CODES))
    recorder = AudioBufferProcessor(num_channels=2)  # left: caller, right: agent
    reply = ReplyTracker()
    latency = LatencyObserver(on_turn)
    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))),
        stt,          # Azure hi-IN ‖ en-IN
        AsrFeed(),    # live partials/finals of both languages to the console
        gate,         # sees both languages' transcripts, so it catches Hindi and English voicemail phrases
        BilingualTranscriber(detector, tracker, on_switch, lid_history),  # merge -> LID -> resolved transcript
        user_agg, llm,
        reply,        # remembers the full intended reply, to compare with what was heard on a barge-in
        tts,
        transport.output(),
        recorder,
        assistant_agg,  # after output: context keeps only the words actually played (Azure word timestamps)
    ])
    worker = PipelineWorker(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency], idle_timeout_secs=120,
        # FULL reports tool arguments and results to the local browser console.
        rtvi_observer_params=RTVIObserverParams(function_call_report_level={'*': RTVIFunctionCallReportLevel.FULL}))
    call_running = True

    @assistant_agg.event_handler('on_assistant_turn_stopped')
    async def on_assistant_turn_stopped(_agg, message):
        """Barge-in: the context already holds only the words played; tell the LLM what the caller missed."""
        if not message.interrupted or not reply.intended:
            return
        note, missed = interruption_note(reply.intended, message.content or '', greeting=reply.intended in GREETING.values())
        if note:
            context.add_message({'role': 'system', 'content': note})
            await worker.queue_frames([server_message('interruption', {'heard': message.content or '', 'missed': missed})])

    @recorder.event_handler('on_audio_data')
    async def on_audio_data(_buf, audio, sample_rate, num_channels):
        path = log.out_dir / f'{log.call_id}.wav'
        log.out_dir.mkdir(parents=True, exist_ok=True)

        def write():
            with wave.open(str(path), 'wb') as w:
                w.setnchannels(num_channels)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio)
        await asyncio.to_thread(write)
        log.data['recording'] = str(path)
        log.data['audio'] = {'duration_s': round(len(audio) / (sample_rate * 2 * num_channels), 2),
                     'sample_rate': sample_rate, 'channels': num_channels, 'format': 'wav'}

    @transport.event_handler('on_client_connected')
    async def on_connected(_transport, _client):
        log.mark_started()
        logger.info(f'CALL_START {log.call_id} lang={flow.lang}')
        await recorder.start_recording()
        gate.start()

    @transport.event_handler('on_client_disconnected')
    async def on_disconnected(_transport, _client):
        await recorder.stop_recording()
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()
    call_running = False
    await latency.flush()
    path = log.save(tracker, context.messages)
    logger.info(f'CALL_END {log.call_id} disposition={log.data["disposition"]} metadata={path} '
                f'latency={log.data["latency_summary"]}')


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, {
        'webrtc': lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)})
    await run_bot(transport, runner_args)


if __name__ == '__main__':
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse
    from pipecat.runner.run import app, main

    recordings_dir = Path(__file__).resolve().parent / 'recordings'

    def require_local(request):
        host = request.client.host if request.client else ''
        if host not in {'127.0.0.1', '::1', 'testclient'}:
            raise HTTPException(status_code=403, detail='Call history is available only from this machine.')

    @app.get('/call', include_in_schema=False)
    async def call_page():
        return FileResponse(os.path.join(os.path.dirname(__file__), 'web', 'call.html'))

    @app.get('/api/calls', include_in_schema=False)
    async def list_calls(request: Request):
        require_local(request)
        calls = []
        for path in recordings_dir.glob('*.json'):
            try:
                call = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            call_id = call.get('call_id', path.stem)
            calls.append({'call_id': call_id, 'started_at': call.get('started_at'),
                          'ended_at': call.get('ended_at'), 'disposition': call.get('disposition'),
                          'summary': call.get('summary', {}), 'latency_summary': call.get('latency_summary', {}),
                          'recording_available': (recordings_dir / f'{path.stem}.wav').is_file()})
        calls.sort(key=lambda call: call.get('started_at') or '', reverse=True)
        return calls

    @app.get('/api/calls/{call_id}', include_in_schema=False)
    async def get_call(call_id: str, request: Request):
        require_local(request)
        if not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{6}', call_id):
            raise HTTPException(status_code=404, detail='Call not found.')
        path = recordings_dir / f'{call_id}.json'
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Call not found.')
        return json.loads(path.read_text())

    @app.get('/api/calls/{call_id}/audio', include_in_schema=False)
    async def get_call_audio(call_id: str, request: Request):
        require_local(request)
        if not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{6}', call_id):
            raise HTTPException(status_code=404, detail='Recording not found.')
        path = recordings_dir / f'{call_id}.wav'
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Recording not found.')
        return FileResponse(path, media_type='audio/wav', filename=path.name)

    main()
