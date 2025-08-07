import io
import os
import re
import time
import requests
import numpy as np
import base64
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
import threading
import queue


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://localhost:8080/v1/tts")
    ref_voice_path: str = Field(default="ref_voice/codes_0.npy")
    sample_rate: int = Field(default=24000)
    streaming: bool = Field(default=False)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.ref_tokens = None
        self.streaming_queue = None
        self.streaming_thread = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_url = None
        self.ref_voice_path = None
        self.ref_tokens = None
        self.sample_rate = None
        self.streaming = False

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
        
        # 加载参考音色的tokens
        try:
            ref_voice_full_path = os.path.join(DirectoryInfo.get_project_dir(), self.ref_voice_path)
            with open(ref_voice_full_path, "rb") as f:
                audio_bytes = f.read()
            self.ref_tokens = base64.b64encode(audio_bytes).decode("utf-8")
            # self.ref_tokens = np.load(ref_voice_full_path).tolist()
            # logger.info(f"Loaded reference voice tokens from {ref_voice_full_path}")
        except Exception as e:
            logger.error(f"Failed to load reference voice tokens: {e}")
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
    
    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        # 测试API连接
        try:
            payload = {
                "text": "こんにちは",
                "references": [{"audio": context.ref_tokens, "text": "こんにちは、私の名前はヒロですね、よろしくお願いますね"}],
                "streaming": self.streaming
            }
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("FishSpeech TTS API connection test successful")
            else:
                logger.warning(f"FishSpeech TTS API test failed with status: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to FishSpeech TTS API: {e}")

    def filter_text(self, text):
        # pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # 匹配不在范围内的字符
        # filtered_text = re.sub(pattern, "", text)
        # return filtered_text
        return text  # 目前不进行过滤，直接返回原文本

    def synthesize_speech_streaming(self, text: str, context: TTSContext, output_definition, speech_id: str):
        """流式语音合成 - 修复版本，完整处理音频文件格式"""
        try:
            payload = {
                "text": text,
                "references": [{"audio": context.ref_tokens, "text": "こんにちは、私の名前はヒロですね、よろしくお願いますね"}],
                "streaming": True
            }
            
            response = requests.post(self.api_url, json=payload, stream=True, timeout=30)
            if response.status_code == 200:
                # 收集完整的流式响应数据
                audio_chunks = []
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        audio_chunks.append(chunk)
                
                if audio_chunks:
                    # 合并所有音频数据
                    complete_audio_data = b''.join(audio_chunks)
                    
                    # 使用与非流式模式相同的解析方式，确保音质一致
                    try:
                        # 优先使用librosa处理，与非流式模式保持一致
                        output_audio = librosa.load(io.BytesIO(complete_audio_data), sr=self.sample_rate)[0]
                        output_audio = output_audio[np.newaxis, ...]
                        
                        # 输出完整的音频
                        output = DataBundle(output_definition)
                        output.set_main_data(output_audio)
                        output.add_meta("avatar_speech_end", False)
                        output.add_meta("speech_id", speech_id)
                        context.submit_data(output)
                        
                    except Exception as e:
                        logger.error(f"Failed to load streaming audio with librosa: {e}")
                        # 备用方案：尝试PCM格式
                        try:
                            output_audio = np.frombuffer(complete_audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                            output_audio = output_audio[np.newaxis, ...]
                            
                            output = DataBundle(output_definition)
                            output.set_main_data(output_audio)
                            output.add_meta("avatar_speech_end", False)
                            output.add_meta("speech_id", speech_id)
                            context.submit_data(output)
                            
                        except Exception as e2:
                            logger.error(f"Failed to parse streaming audio as PCM: {e2}")
                else:
                    logger.warning("Received empty streaming response")
                        
            else:
                logger.error(f"FishSpeech streaming API failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")

    def synthesize_speech(self, text: str, context: TTSContext) -> Optional[np.ndarray]:
        """调用FishSpeech API生成语音"""
        try:
            payload = {
                "text": text,
                "references": [{"audio": context.ref_tokens, "text": "こんにちは、私の名前はヒロですね、よろしくお願いますね"}],
                "streaming": self.streaming
            }
            
            # 非流式模式
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                audio_data = response.content
                try:
                    output_audio = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)[0]
                    output_audio = output_audio[np.newaxis, ...]
                    return output_audio
                except Exception as e:
                    logger.error(f"Failed to load audio with librosa: {e}")
                    # 尝试其他解析方式
                    try:
                        output_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                        output_audio = output_audio[np.newaxis, ...]
                        return output_audio
                    except Exception as e2:
                        logger.error(f"Failed to parse as PCM data: {e2}")
                        return None
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
                        # 流式模式：异步处理以实现实时输出
                        def stream_process():
                            self.synthesize_speech_streaming(sentence.strip(), context, output_definition, speech_id)
                        
                        stream_thread = threading.Thread(target=stream_process)
                        stream_thread.start()
                    else:
                        # 非流式模式
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
                    # 流式模式：处理最后一句
                    def stream_final_process():
                        self.synthesize_speech_streaming(context.input_text.strip(), context, output_definition, speech_id)
                    
                    stream_thread = threading.Thread(target=stream_final_process)
                    stream_thread.start()
                else:
                    # 非流式模式
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