from dataclasses import Field, dataclass, field
import logging
from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp
import torch
import torchaudio
from datetime import datetime

import os
import sys

import librosa
from loguru import logger
import numpy as np
import requests

from engine_utils.directory_info import DirectoryInfo


# @dataclass
# class TTSConfig:
#     model: any = None
#     model_name: str = field(default=None)
#     api_url: str = field(default=None)
#     spk_id: str = field(default=None)
#     ref_audio_text: str = field(default=None)
#     ref_audio_buffer: np.ndarray = None
#     sample_rate: int = field(default=24000)
spawn_context = mp.get_context('spawn')

class TTSCosyVoiceProcessor(spawn_context.Process):
    def __init__(self, handler_root: str, config: any, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.handler_root = handler_root
        self.model = None
        self.model_name = config.model_name
        self.api_url = config.api_url
        self.spk_id = config.spk_id
        self.ref_audio_text = config.ref_audio_text
        self.ref_audio_path = config.ref_audio_path
        self.ref_audio_buffer = None
        self.sample_rate = config.sample_rate
        self.api_key = config.api_key
        self.save_audio_path = "./tts_audio"

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.dump_audio = False

    def run(self):
        logger.remove()
        logger.add(sys.stdout, level='INFO')
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(),
                                            "dump_avatar_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        logger.info('start tts processor')
        # use local model
        if self.api_key is None and self.model_name is not None:
            sys.path.append(os.path.join(self.handler_root, "CosyVoice"))
            sys.path.append(os.path.join(self.handler_root, 'CosyVoice', 'third_party', 'Matcha-TTS'))
            from handlers.tts.cosyvoice.CosyVoice.cosyvoice.utils.file_utils import load_wav
            from handlers.tts.cosyvoice.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
            try:
                self.model = CosyVoice2(model_dir=self.model_name, load_jit=True, load_trt=True, fp16=True)
            except Exception:
                try:
                    self.model = CosyVoice(model_dir=self.model_name)
                except Exception:
                    raise TypeError('no valid model_type!')
            self.model.sample_rate = self.sample_rate
            if self.ref_audio_path:
                self.ref_audio_buffer = load_wav(self.ref_audio_path, self.sample_rate)
                self.ref_audio_text = self.ref_audio_text
            init_text = "おはようございます"
            response = None
            if self.ref_audio_buffer is not None:
                response = self.model.inference_zero_shot(
                    init_text, self.ref_audio_text, self.ref_audio_buffer, stream=True)
            elif self.spk_id:
                response = self.model.inference_instruct2(init_text, self.spk_id, self.ref_audio_buffer, stream=True)
            else:
                logger.error('cosyvoice need a ref_audio or spk_id')
                return
            if response is not None:
                for _ in response:
                    pass  # Consume generator for warmup
            self.output_queue.put({'key': 'cosyvoice_ready', 'tts_speech': None, 'session_id': None})
        elif self.api_key is not None:
            raise TypeError('api_key not support yet')
        logger.info('tts processor started')
        while True:
            try:
                logger.debug('wait for tts task in')
                input = self.input_queue.get(timeout=5)
                logger.debug(f'get tts task in {input}')
            except Exception:
                continue
            input_text = input['text']
            key = input['key']
            session_id = input['session_id']
            if (len(input_text) < 1):
                # ignore
                logger.info('ignore empty input_text')
            elif self.model is None and self.api_url is not None:
                # if you start cosyvoice tts server through CosyVoice/runtime/python/fastapi/server.py
                response = requests.get(self.api_url, data={
                    'tts_text': input_text,
                    'spk_id': self.spk_id
                }, stream=True)
                if response.status_code != 200:
                    logger.info(f"Request failed with status code {response.status_code}")
                    continue
                tts_audio = b''
                for r in response.iter_content(chunk_size=16000):
                    tts_audio = r
                    tts_speech = np.array(np.frombuffer(tts_audio, dtype=np.int16)).astype(np.float32)/32767
                    logger.debug(f'audio response {tts_speech.shape}')

                    output_audio = librosa.resample(tts_speech, orig_sr=22050, target_sr=self.sample_rate)
                    logger.debug(f'audio response resample {output_audio.shape}')
                    out_audio = output_audio[np.newaxis, ...]
                    output = {
                        'key': key,
                        'tts_speech': out_audio,
                        'session_id': session_id
                    }
                    self.output_queue.put(output)
            # if self.api_key is not None:
            #     self.model.streaming_call(input_text)

            #     for tts_audio in self.callback_instance.get_data_generator():
            #         tts_speech = np.array(np.frombuffer(tts_audio, dtype=np.int16)).astype(np.float32)/32767
            #         logger.info('audio response', tts_speech.shape)

            #         output_audio = librosa.resample(tts_speech, orig_sr=self.sample_rate, target_sr=24000)
            #         out_audio = output_audio[np.newaxis, ...]
            #         yield out_audio
            else:
                response = None
                if self.model:
                    if self.ref_audio_buffer is not None:
                        logger.info(f"================ {input_text}")
                        response = self.model.inference_zero_shot(
                            input_text, self.ref_audio_text, self.ref_audio_buffer, stream=True)
                    elif self.spk_id:
                        response = self.model.inference_instruct2(input_text, self.spk_id, self.ref_audio_buffer, stream=True)
                    else:
                        logger.error('cosyvoice need a ref_audio or spk_id')
                        return

                if response:
                    audio_chunks = []
                    for i, tts_speech in enumerate(response):
                        tts_audio = tts_speech['tts_speech']
                        tts_audio_numpy = tts_audio.numpy()
                        audio_chunks.append(tts_audio_numpy)
                        logger.debug(f'tts sample rate {self.model.sample_rate}')
                        
                        if self.dump_audio:
                            dump_audio = tts_audio_numpy
                            self.audio_dump_file.write(dump_audio.tobytes())
                        output = {
                            'key': key,
                            'tts_speech': tts_audio_numpy,
                            'session_id': session_id
                        }
                        self.output_queue.put(output)

                    if self.save_audio_path and audio_chunks:
                        full_audio_numpy = np.concatenate(audio_chunks, axis=1)
                        full_audio_tensor = torch.from_numpy(full_audio_numpy)
                        
                        os.makedirs(self.save_audio_path, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"tts_{timestamp}.wav"
                        save_path = os.path.join(self.save_audio_path, filename)
                        
                        torchaudio.save(save_path, full_audio_tensor, self.model.sample_rate)
                        logger.info(f"Saved synthesized audio to {save_path}")
            output = {
                'key': key,
                'tts_speech': None,
                'session_id': session_id
            }
            self.output_queue.put(output)
