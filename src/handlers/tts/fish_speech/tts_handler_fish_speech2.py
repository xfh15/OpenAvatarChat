import io
import os
import re
import time
import asyncio
import numpy as np
from typing import Dict, Optional, cast, AsyncGenerator
import librosa
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo

try:
    from fish_audio_sdk import AsyncWebSocketSession, TTSRequest, ReferenceAudio
except ImportError:
    AsyncWebSocketSession = None
    TTSRequest = None
    ReferenceAudio = None
    logger.error("fish_audio_sdk not installed. Please install it with: pip install fish-audio-sdk")


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    api_key: str = Field(default=os.getenv("FISH_API_KEY"))
    ref_voice_path: str = Field(default="ref_voice/ref.wav")
    ref_voice_text: str = Field(default="こんにちは、私の名前はヒロですね、よろしくお願いますね")
    sample_rate: int = Field(default=24000)
    streaming: bool = Field(default=True)
    # Speech generation parameters
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.7)
    backend: str = Field(default="speech-1.6")  # speech-1.5, speech-1.6, s1
    # Optional: use reference_id instead of reference audio
    reference_id: Optional[str] = Field(default=None)
    audio_format: str = Field(default="mp3")  # Output format from API


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.ref_audio_bytes = None
        self.websocket_session = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_key = None
        self.ref_voice_path = None
        self.ref_voice_text = None
        self.ref_audio_bytes = None
        self.sample_rate = None
        self.streaming = False
        self.temperature = 0.7
        self.top_p = 0.7
        self.backend = "speech-1.6"
        self.reference_id = None
        self.audio_format = "mp3"

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if AsyncWebSocketSession is None:
            raise RuntimeError("fish_audio_sdk not installed. Please install it with: pip install fish-audio-sdk")
        
        config = cast(TTSConfig, handler_config)
        self.api_key = config.api_key
        self.ref_voice_path = config.ref_voice_path
        self.ref_voice_text = config.ref_voice_text
        self.sample_rate = config.sample_rate
        self.streaming = config.streaming
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.backend = config.backend
        self.reference_id = config.reference_id
        self.audio_format = config.audio_format
        
        if not self.api_key:
            raise ValueError("API key is required for Fish Speech official API")
        
        # Load reference audio if reference_id is not provided
        if not self.reference_id:
            try:
                ref_voice_full_path = os.path.join(DirectoryInfo.get_project_dir(), self.ref_voice_path)
                with open(ref_voice_full_path, "rb") as f:
                    self.ref_audio_bytes = f.read()
                logger.info(f"Loaded reference audio from: {ref_voice_full_path}")
            except Exception as e:
                logger.error(f"Failed to load reference voice: {e}")
                raise

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        context.ref_audio_bytes = self.ref_audio_bytes
        context.websocket_session = AsyncWebSocketSession(self.api_key)
        
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                        f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context

    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info("Fish Speech official API TTS handler initialized")

    def filter_text(self, text):
        # Keep original logic: no filtering
        return text

    def _create_tts_request(self, context: TTSContext) -> TTSRequest:
        """Create TTSRequest object based on configuration."""
        if self.reference_id:
            # Use reference_id
            return TTSRequest(
                text="",
                reference_id=self.reference_id,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            # Use reference audio
            return TTSRequest(
                text="",
                references=[
                    ReferenceAudio(
                        audio=context.ref_audio_bytes,
                        text=self.ref_voice_text,
                    )
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )

    async def _text_stream_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Generator to stream text word by word."""
        words = text.split()
        for word in words:
            yield word + " "

    def _submit_chunk(self, audio_bytes: bytes, context: TTSContext, output_definition, speech_id):
        """Convert audio bytes to numpy array and submit as DataBundle."""
        if not audio_bytes:
            return
        
        try:
            # Load audio from bytes and resample to target sample rate
            audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
            if audio_data.size == 0:
                return
            
            output_audio = audio_data[np.newaxis, ...]
            output = DataBundle(output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
        except Exception as e:
            logger.error(f"Failed to submit chunk: {e}")

    async def _synthesize_streaming(self, text: str, context: TTSContext, output_definition, speech_id):
        """Stream synthesis using Fish Speech WebSocket API."""
        try:
            tts_request = self._create_tts_request(context)
            
            # Collect audio chunks
            audio_chunks = []
            async for chunk in context.websocket_session.tts(
                tts_request,
                self._text_stream_generator(text),
                backend=self.backend
            ):
                audio_chunks.append(chunk)
                
                # Submit chunks periodically to maintain streaming feel
                if len(audio_chunks) >= 5:  # Adjust chunk size as needed
                    combined_chunk = b''.join(audio_chunks)
                    self._submit_chunk(combined_chunk, context, output_definition, speech_id)
                    audio_chunks = []
            
            # Submit remaining chunks
            if audio_chunks:
                combined_chunk = b''.join(audio_chunks)
                self._submit_chunk(combined_chunk, context, output_definition, speech_id)
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")

    async def _synthesize_non_streaming(self, text: str, context: TTSContext) -> Optional[np.ndarray]:
        """Non-streaming synthesis using Fish Speech WebSocket API."""
        try:
            tts_request = self._create_tts_request(context)
            
            # Collect all audio chunks
            audio_chunks = []
            async for chunk in context.websocket_session.tts(
                tts_request,
                self._text_stream_generator(text),
                backend=self.backend
            ):
                audio_chunks.append(chunk)
            
            if audio_chunks:
                # Combine all chunks
                combined_audio = b''.join(audio_chunks)
                # Load and resample audio
                audio_data, _ = librosa.load(io.BytesIO(combined_audio), sr=self.sample_rate)
                output_audio = audio_data[np.newaxis, ...]
                return output_audio
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in non-streaming synthesis: {e}")
            return None

    def _run_async_synthesis(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use asyncio.create_task
                # But since we're in sync context, we'll use run_until_complete
                pass
        except RuntimeError:
            loop = None
        
        if loop is None or loop.is_closed():
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        else:
            return loop.run_until_complete(coro)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
            
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:  # At least one complete sentence
                complete_sentences = sentences[:-1]  # Complete sentences
                context.input_text = sentences[-1]  # Remaining incomplete part

                # Process complete sentences
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    logger.info(f'Processing sentence: {sentence}')
                    if self.streaming:
                        coro = self._synthesize_streaming(sentence.strip(), context, output_definition, speech_id)
                        self._run_async_synthesis(coro)
                    else:
                        coro = self._synthesize_non_streaming(sentence.strip(), context)
                        output_audio = self._run_async_synthesis(coro)
                        if output_audio is not None:
                            output = DataBundle(output_definition)
                            output.set_main_data(output_audio)
                            output.add_meta("avatar_speech_end", False)
                            output.add_meta("speech_id", speech_id)
                            context.submit_data(output)
        else:
            logger.info(f'Processing last sentence: {context.input_text}')
            if context.input_text is not None and len(context.input_text.strip()) > 0:
                if self.streaming:
                    coro = self._synthesize_streaming(context.input_text.strip(), context, output_definition, speech_id)
                    self._run_async_synthesis(coro)
                else:
                    coro = self._synthesize_non_streaming(context.input_text.strip(), context)
                    output_audio = self._run_async_synthesis(coro)
                    if output_audio is not None:
                        output = DataBundle(output_definition)
                        output.set_main_data(output_audio)
                        output.add_meta("avatar_speech_end", False)
                        output.add_meta("speech_id", speech_id)
                        context.submit_data(output)
                    
            context.input_text = ''
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, self.sample_rate), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            logger.info(f"Speech synthesis completed")

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        if hasattr(context, 'websocket_session') and context.websocket_session:
            # Close WebSocket session if needed
            try:
                # The SDK should handle cleanup automatically
                context.websocket_session = None
            except Exception as e:
                logger.warning(f"Error closing WebSocket session: {e}")
        logger.info('Destroying Fish Speech official API TTS context')
