import io
import os
import re
import time
import requests
import numpy as np
from typing import Dict, Optional, cast
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


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://localhost:8080/v1/tts")
    ref_voice_path: str = Field(default="ref_voice/ref.wav")
    sample_rate: int = Field(default=24000)
    streaming: bool = Field(default=True)
    # Additional options to match api_client behavior
    audio_format: str = Field(default="wav")  # wav | mp3 | flac
    channels: int = Field(default=1)
    api_key: Optional[str] = Field(default=os.getenv("FISH_API_KEY"))
    source_sample_rate: int = Field(default=44100)  # 服务端流式输出的采样率


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.ref_tokens = None  # will store raw bytes of reference audio


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_url = None
        self.ref_voice_path = None
        self.ref_tokens = None
        self.sample_rate = None
        self.streaming = False
        # New fields for FishSpeech streaming
        self.audio_format = "wav"
        self.channels = 1
        self.api_key: Optional[str] = None
        self.source_sample_rate: int = 44100

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
        config = cast(TTSConfig, handler_config)
        self.api_url = config.api_url
        self.ref_voice_path = config.ref_voice_path
        self.sample_rate = config.sample_rate
        self.streaming = config.streaming
        self.audio_format = getattr(config, 'audio_format', 'wav')
        self.channels = getattr(config, 'channels', 1)
        self.api_key = getattr(config, 'api_key', None)
        self.source_sample_rate = getattr(config, 'source_sample_rate', 44100)
        
        # 加载参考音色的原始音频字节（与 api_client 对齐）
        try:
            ref_voice_full_path = os.path.join(DirectoryInfo.get_project_dir(), self.ref_voice_path)
            with open(ref_voice_full_path, "rb") as f:
                audio_bytes = f.read()
            # 对 FishSpeech 的 msgpack API，"audio" 字段应为原始字节
            self.ref_tokens = audio_bytes
        except Exception as e:
            logger.error(f"Failed to load reference voice: {e}")
            raise

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        context.ref_tokens = self.ref_tokens
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                        f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context

    def _pack_payload(self, payload: dict) -> tuple[bytes, str]:
        """Pack payload into MsgPack if available; otherwise raise for clear guidance."""
        try:
            import ormsgpack as _mp
            return _mp.packb(payload), "application/msgpack"
        except Exception:
            try:
                import msgpack as _mp
                return _mp.packb(payload, use_bin_type=True), "application/msgpack"
            except Exception:
                raise RuntimeError("Neither 'ormsgpack' nor 'msgpack' is installed. Please install one to use FishSpeech streaming.")

    def _headers(self, content_type: str):
        headers = {"content-type": content_type}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_request_payload(self, text: str, context: TTSContext, streaming: bool):
        # 与 api_client.py 的字段对齐
        return {
            "text": text,
            "references": [{
                "audio": context.ref_tokens if context.ref_tokens is not None else b"",
                "text": "こんにちは、私の名前はヒロですね、よろしくお願いますね",
            }],
            "reference_id": None,
            "format": self.audio_format,
            "max_new_tokens": 1024,
            "chunk_length": 300,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "temperature": 0.8,
            "streaming": streaming,
            "use_memory_cache": "off",
            "seed": None,
        }
    
    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        # 测试 API 连接（msgpack 请求）
        try:
            payload = self._build_request_payload("こんにちは", context, streaming=False)
            body, ctype = self._pack_payload(payload)
            response = requests.post(self.api_url, data=body, timeout=10, headers=self._headers(ctype))
            if response.status_code == 200:
                logger.info("FishSpeech TTS API connection test successful")
            else:
                logger.warning(f"FishSpeech TTS API test failed with status: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to FishSpeech TTS API: {e}")

    def filter_text(self, text):
        # 保持原有逻辑：不做过滤
        return text

    def _submit_chunk(self, pcm_bytes: bytes, context: TTSContext, output_definition, speech_id):
        if not pcm_bytes:
            return
        # 将 int16 PCM 转 float32 [-1, 1]，并从源采样率重采样到目标采样率
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            if samples.size == 0:
                return
            if self.sample_rate and self.source_sample_rate and self.sample_rate != self.source_sample_rate:
                # 重采样到目标采样率（例如 44100 -> 24000）
                samples = librosa.resample(samples, orig_sr=self.source_sample_rate, target_sr=self.sample_rate)
            output_audio = samples[np.newaxis, ...]
            output = DataBundle(output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
        except Exception as e:
            logger.error(f"Failed to submit chunk: {e}")

    def _synthesize_streaming(self, text: str, context: TTSContext, output_definition, speech_id):
        """流式调用 FishSpeech，并边收边推送 DataBundle。"""
        try:
            payload = self._build_request_payload(text, context, streaming=True)
            body, ctype = self._pack_payload(payload)
            response = requests.post(
                self.api_url,
                data=body,
                stream=True,
                timeout=60,
                headers=self._headers(ctype),
            )
            if response.status_code != 200:
                logger.error(f"FishSpeech streaming API request failed with status {response.status_code}: {response.text}")
                return

            buffer = bytearray()
            # 以源采样率估算约 0.5 秒的块大小：int16 两字节 -> 0.5s 样本数 = source_sr/2，对应字节数约为 source_sr
            threshold_bytes = max(int(self.source_sample_rate), 4096)

            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                buffer.extend(chunk)
                # 保证偶数字节（int16 对齐）
                while len(buffer) >= threshold_bytes:
                    flush_size = (threshold_bytes // 2) * 2
                    pcm = bytes(buffer[:flush_size])
                    del buffer[:flush_size]
                    self._submit_chunk(pcm, context, output_definition, speech_id)

            # 剩余不足阈值的也输出一次
            if len(buffer) > 0:
                flush_size = (len(buffer) // 2) * 2
                if flush_size > 0:
                    self._submit_chunk(bytes(buffer[:flush_size]), context, output_definition, speech_id)
        except Exception as e:
            logger.error(f"Error calling FishSpeech streaming API: {e}")

    def synthesize_speech(self, text: str, context: TTSContext) -> Optional[np.ndarray]:
        """非流式：调用 FishSpeech API 生成整段语音并返回。"""
        try:
            payload = self._build_request_payload(text, context, streaming=False)
            body, ctype = self._pack_payload(payload)
            response = requests.post(self.api_url, data=body, timeout=60, headers=self._headers(ctype))
            if response.status_code == 200:
                audio_data = response.content
                # librosa 可读取多种格式（wav/mp3/flac），重采样到 self.sample_rate
                output_audio = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)[0]
                output_audio = output_audio[np.newaxis, ...]
                return output_audio
            else:
                logger.error(f"FishSpeech API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling FishSpeech API: {e}")
            return None

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
            if len(sentences) > 1:  # 至少有一个完整句子
                complete_sentences = sentences[:-1]  # 完整句子
                context.input_text = sentences[-1]  # 剩余的未完成部分

                # 对完整句子进行处理
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    logger.info(f'Processing sentence: {sentence}')
                    if self.streaming:
                        self._synthesize_streaming(sentence.strip(), context, output_definition, speech_id)
                    else:
                        output_audio = self.synthesize_speech(sentence.strip(), context)
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
                    self._synthesize_streaming(context.input_text.strip(), context, output_definition, speech_id)
                else:
                    output_audio = self.synthesize_speech(context.input_text.strip(), context)
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
        logger.info('Destroying FishSpeech TTS context')