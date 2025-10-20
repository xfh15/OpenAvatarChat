import os
import queue
import threading
import time
from queue import Queue
from threading import Thread
from typing import Optional
from collections import deque # 新增导入

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from handlers.avatar.liteavatar.model.algo_model import AvatarStatus, AudioResult, VideoResult
from handlers.avatar.liteavatar.model.audio_input import SpeechAudio
from src.handlers.avatar.musetalk.avatar_musetalk_algo import MuseAvatarV15
from src.handlers.avatar.musetalk.avatar_musetalk_config import AvatarMuseTalkConfig

class AvatarMuseTalkProcessor:
    """MuseTalk processor responsible for audio-to-video conversion (multi-threaded queue structure)."""
    
    def __init__(self, avatar: MuseAvatarV15, config: AvatarMuseTalkConfig):
        self._avatar = avatar
        self._config = config
        self._algo_audio_sample_rate = config.algo_audio_sample_rate  # Internal algorithm sample rate, fixed at 16000
        self._output_audio_sample_rate = config.output_audio_sample_rate
        # Output queues
        self.audio_output_queue = None
        self.video_output_queue = None
        self.event_out_queue = None
        # Internal queues
        self._audio_queue = Queue()  # Input audio queue
        self._whisper_queue = Queue()  # Whisper feature queue
        self._unet_queue = Queue()  # Unet output queue
        # self._frame_queue = Queue() # 该队列在多线程VAEvsUNET版本中未使用，予以注释
        # self._frame_id_queue = Queue() # [修改] 弃用，由新的同步机制替代
        self._compose_queue = Queue()  # Frame composition queue
        self._output_queue = Queue()   # Output queue after composition
        # Threading and state
        self._stop_event = threading.Event()
        self._feature_thread: Optional[Thread] = None
        self._frame_gen_thread: Optional[Thread] = None
        self._frame_gen_unet_thread: Optional[Thread] = None
        self._frame_gen_vae_thread: Optional[Thread] = None
        self._frame_collect_thread: Optional[Thread] = None
        self._compose_thread: Optional[Thread] = None
        self._session_running = False
        # Avatar status
        self._callback_avatar_status = AvatarStatus.LISTENING
        self._last_speech_id = None
        # Audio duration statistics
        self._first_add_audio_time = None
        self._audio_duration_sum = 0.0
        # Audio cache for each speech_id
        self._audio_cache = {}

        # [新增] 延迟补偿与同步机制所需变量
        self._pipeline_latency_ms = 150.0  # 初始延迟估计值(ms)，可根据设备性能调整
        self._latency_measurements = deque(maxlen=20)  # 存储最近20次延迟测量值
        self._frame_generation_timestamps = {}  # 记录每个帧ID的生成请求时间戳
        self._current_collector_frame_id = 0  # 播放器当前的最新帧ID，由Collector线程唯一写入
        self._frame_id_lock = threading.Lock() # 用于保护时间戳字典等共享资源

    def start(self):
        """Start the processor and all worker threads."""
        if self._session_running:
            logger.error("Processor already running. session_running=True")
            return
        self._session_running = True
        self._stop_event.clear()
        try:
            self._feature_thread = threading.Thread(target=self._feature_extractor_worker)
            if self._config.multi_thread_inference:
                self._frame_gen_unet_thread = threading.Thread(target=self._frame_generator_unet_worker)
                self._frame_gen_vae_thread = threading.Thread(target=self._frame_generator_vae_worker)
            else:
                self._frame_gen_thread = threading.Thread(target=self._frame_generator_worker)
            self._frame_collect_thread = threading.Thread(target=self._frame_collector_worker)
            self._compose_thread = threading.Thread(target=self._compose_worker)
            self._feature_thread.start()
            if self._config.multi_thread_inference:
                self._frame_gen_unet_thread.start()
                self._frame_gen_vae_thread.start()
            else:
                self._frame_gen_thread.start()
            self._frame_collect_thread.start()
            self._compose_thread.start()
            logger.info(f"MuseProcessor started.")
        except Exception as e:
            logger.opt(exception=True).error(f"Exception during thread start: {e}")

    def stop(self):
        """Stop the processor and all worker threads."""
        if not self._session_running:
            logger.warning("Processor not running. Skip stop.")
            return
        self._session_running = False
        self._stop_event.set()
        try:
            if self._feature_thread: self._feature_thread.join(timeout=5)
            if self._frame_gen_thread: self._frame_gen_thread.join(timeout=5)
            if self._frame_gen_unet_thread: self._frame_gen_unet_thread.join(timeout=5)
            if self._frame_gen_vae_thread: self._frame_gen_vae_thread.join(timeout=5)
            if self._frame_collect_thread: self._frame_collect_thread.join(timeout=5)
            if self._compose_thread: self._compose_thread.join(timeout=5)
            self._clear_queues()
        except Exception as e:
            logger.opt(exception=True).error(f"Exception during thread join: {e}")
        logger.info(f"MuseProcessor stopped.")

    def add_audio(self, speech_audio: SpeechAudio):
        """
        Add an audio segment to the processing queue. The segment length must not exceed 1 second. No resampling is performed here.
        Args:
            speech_audio (SpeechAudio): Audio segment to add.
        """
        if self._config.debug:
            now = time.time()
            if self._first_add_audio_time is None: self._first_add_audio_time = now
            audio_len = len(speech_audio.audio_data)
            sample_rate = speech_audio.sample_rate
            audio_duration = audio_len / 4 / sample_rate
            self._audio_duration_sum += audio_duration
            total_interval = now - self._first_add_audio_time
            log_msg = (
                f"Received add_audio: speech_id={speech_audio.speech_id}, eos={speech_audio.end_of_speech}, "
                f"duration={audio_duration:.3f}s, cumulative_audio={self._audio_duration_sum:.3f}s, total_interval={total_interval:.3f}s"
            )
            logger.info(log_msg)
            if speech_audio.end_of_speech:
                self._audio_duration_sum = 0.0
                self._first_add_audio_time = None

        audio_data = np.frombuffer(speech_audio.audio_data, dtype=np.float32) if isinstance(speech_audio.audio_data, bytes) else speech_audio.audio_data.astype(np.float32)
        
        if len(audio_data) == 0:
            logger.error(f"Input audio is empty, speech_id={speech_audio.speech_id}")
            return
        
        try:
            self._audio_queue.put({
                'audio_data': audio_data,
                'speech_id': speech_audio.speech_id,
                'end_of_speech': speech_audio.end_of_speech,
            }, timeout=1)
        except queue.Full:
            logger.opt(exception=True).error(f"Audio queue full, dropping audio segment, speech_id={speech_audio.speech_id}")

    def _feature_extractor_worker(self):
        if torch.cuda.is_available():
            t0 = time.time()
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self._avatar.extract_whisper_feature(dummy_audio, 16000)
            torch.cuda.synchronize()
            logger.info(f"[THREAD_WARMUP] _feature_extractor_worker whisper warmup done, time: {(time.time()-t0)*1000:.1f} ms")
        
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=1)
                audio_data, speech_id, end_of_speech = item['audio_data'], item['speech_id'], item['end_of_speech']
                fps = self._config.fps
                
                segment = librosa.resample(audio_data, orig_sr=self._output_audio_sample_rate, target_sr=self._algo_audio_sample_rate)
                
                target_len = self._algo_audio_sample_rate
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)), mode='constant')
                
                whisper_chunks = self._avatar.extract_whisper_feature(segment, self._algo_audio_sample_rate)
                
                orig_samples_per_frame = self._output_audio_sample_rate // fps
                num_frames = int(np.ceil(len(audio_data) / orig_samples_per_frame))
                whisper_chunks = whisper_chunks[:num_frames]

                target_audio_len = num_frames * orig_samples_per_frame
                if len(audio_data) < target_audio_len:
                    audio_data = np.pad(audio_data, (0, target_audio_len - len(audio_data)), mode='constant')
                else:
                    audio_data = audio_data[:target_audio_len]
                
                num_chunks = len(whisper_chunks)
                for i in range(num_chunks):
                    start_sample = i * orig_samples_per_frame
                    end_sample = start_sample + orig_samples_per_frame
                    audio_segment = audio_data[start_sample:end_sample]
                    
                    self._whisper_queue.put({
                        'whisper_chunks': whisper_chunks[i:i+1],
                        'speech_id': speech_id,
                        'end_of_speech': end_of_speech and (i == num_chunks - 1),
                        'audio_data': audio_segment,
                    }, timeout=1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _feature_extractor_worker: {e}")

    def _frame_generator_unet_worker(self):
        batch_size = self._config.batch_size
        batch_chunks, batch_audio, batch_speech_id, batch_end_of_speech = [], [], [], []
        
        while not self._stop_event.is_set():
            try:
                # [修改] 预测性地获取帧ID
                # 1. 根据当前平均延迟，计算需要提前多少帧 (加2是为了安全缓冲)
                delay_in_frames = int(self._pipeline_latency_ms / 1000 * self._config.fps) + 2
                # 2. 计算我们应该从哪个未来的帧ID开始生成
                start_id = self._current_collector_frame_id + delay_in_frames
                # 3. 准备好这一批次要生成的所有帧的ID
                frame_ids = [start_id + i for i in range(batch_size)]

                item = self._whisper_queue.get(timeout=1)
                batch_chunks.append(item['whisper_chunks'])
                batch_audio.append(item['audio_data'])
                batch_speech_id.append(item['speech_id'])
                batch_end_of_speech.append(item['end_of_speech'])
                
                if len(batch_chunks) == batch_size or item['end_of_speech']:
                    valid_num = len(batch_chunks)
                    if valid_num < batch_size:
                        pad_num = batch_size - valid_num
                        batch_chunks.extend([torch.zeros_like(batch_chunks[0])] * pad_num)
                        batch_audio.extend([np.zeros_like(batch_audio[0])] * pad_num)
                        batch_speech_id.extend([batch_speech_id[-1]] * pad_num)
                        batch_end_of_speech.extend([False] * pad_num)

                    # 4. 记录下我们是什么时候开始为这些帧工作的
                    with self._frame_id_lock:
                        now = time.perf_counter()
                        for i in range(valid_num):
                            self._frame_generation_timestamps[frame_ids[i]] = now
                    
                    whisper_batch = torch.cat(batch_chunks, dim=0)
                    pred_latents, idx_list = self._avatar.generate_frames_unet(whisper_batch, frame_ids[0], batch_size)
                    
                    self._unet_queue.put({
                        'pred_latents': pred_latents,
                        'speech_id': batch_speech_id,
                        'end_of_speech': batch_end_of_speech,
                        'audio_data': batch_audio,
                        'valid_num': valid_num,
                        'idx_list': idx_list, # 这里的idx_list应为我们计算的frame_ids
                    })
                    batch_chunks, batch_audio, batch_speech_id, batch_end_of_speech = [], [], [], []
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _frame_generator_unet_worker: {e}")

    def _frame_generator_vae_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._unet_queue.get(timeout=1)
                pred_latents = item['pred_latents']
                idx_list = item['idx_list'] # 这是由Unet worker传递过来的帧ID列表
                batch_audio = item['audio_data']
                valid_num = item['valid_num']
                batch_speech_id = item['speech_id']
                batch_end_of_speech = item['end_of_speech']
                cur_batch = pred_latents.shape[0]

                recon_idx_list = self._avatar.generate_frames_vae(pred_latents, idx_list, cur_batch)
                 
                for i in range(valid_num):
                    recon, idx = recon_idx_list[i]
                    self._compose_queue.put({
                        'recon': recon,
                        'frame_id': idx, # 直接使用传递过来的帧ID
                        'speech_id': batch_speech_id[i],
                        'avatar_status': AvatarStatus.SPEAKING,
                        'end_of_speech': batch_end_of_speech[i],
                        'audio_segment': batch_audio[i],
                    })
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _frame_generator_vae_worker: {e}")

    def _frame_generator_worker(self):
        """非分离UNET和VAE的单线程生成器，同样需要修改"""
        batch_size = self._config.batch_size
        batch_chunks, batch_audio, batch_speech_id, batch_end_of_speech = [], [], [], []

        while not self._stop_event.is_set():
            try:
                # [修改] 预测性地获取帧ID (逻辑同unet_worker)
                delay_in_frames = int(self._pipeline_latency_ms / 1000 * self._config.fps) + 2
                start_id = self._current_collector_frame_id + delay_in_frames
                frame_ids = [start_id + i for i in range(batch_size)]

                item = self._whisper_queue.get(timeout=1)
                batch_chunks.append(item['whisper_chunks'])
                batch_audio.append(item['audio_data'])
                batch_speech_id.append(item['speech_id'])
                batch_end_of_speech.append(item['end_of_speech'])

                if len(batch_chunks) == batch_size or item['end_of_speech']:
                    valid_num = len(batch_chunks)
                    if valid_num < batch_size:
                        pad_num = batch_size - valid_num
                        batch_chunks.extend([torch.zeros_like(batch_chunks[0])] * pad_num)
                        batch_audio.extend([np.zeros_like(batch_audio[0])] * pad_num)
                        batch_speech_id.extend([batch_speech_id[-1]] * pad_num)
                        batch_end_of_speech.extend([False] * pad_num)
                    
                    with self._frame_id_lock:
                        now = time.perf_counter()
                        for i in range(valid_num):
                            self._frame_generation_timestamps[frame_ids[i]] = now

                    whisper_batch = torch.cat(batch_chunks, dim=0)
                    recon_idx_list = self._avatar.generate_frames(whisper_batch, frame_ids[0], batch_size)

                    for i in range(valid_num):
                        recon, idx = recon_idx_list[i]
                        self._compose_queue.put({
                            'recon': recon,
                            'frame_id': idx,
                            'speech_id': batch_speech_id[i],
                            'avatar_status': AvatarStatus.SPEAKING,
                            'end_of_speech': batch_end_of_speech[i],
                            'audio_segment': batch_audio[i],
                        })
                    batch_chunks, batch_audio, batch_speech_id, batch_end_of_speech = [], [], [], []
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _frame_generator_worker: {e}")

    def _compose_worker(self):
        """将合成结果放入_output_queue"""
        while not self._stop_event.is_set():
            try:
                item = self._compose_queue.get(timeout=0.1)
                recon = item['recon']
                idx = item['frame_id']
                frame = self._avatar.res2combined(recon, idx)
                item['frame'] = frame
                self._output_queue.put(item)
            except queue.Empty:
                continue

    def _frame_collector_worker(self):
        """[核心修改] 精准的帧调度中心"""
        fps = self._config.fps
        frame_interval = 1.0 / fps
        start_time = time.perf_counter()
        local_frame_id = 0
        last_active_speech_id = None
        current_speech_id = None
        
        while not self._stop_event.is_set():
            target_time = start_time + local_frame_id * frame_interval
            
            # 精确的sleep以控制FPS
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_delivery_time = time.perf_counter()
            
            # [修改] 实时更新全局播放器进度
            self._current_collector_frame_id = local_frame_id
            
            output_item = None
            future_items = []
            
            # 从输出队列中寻找与当前时间点匹配的帧
            while not self._output_queue.empty():
                try:
                    item = self._output_queue.get_nowait()
                    item_frame_id = item.get('frame_id')

                    if item_frame_id == local_frame_id:
                        output_item = item
                        break # 找到了完美匹配的帧
                    elif item_frame_id < local_frame_id:
                        logger.warning(f"Dropping stale frame. Player is at {local_frame_id}, but received {item_frame_id}.")
                        with self._frame_id_lock: # 清理时间戳记录
                           self._frame_generation_timestamps.pop(item_frame_id, None)
                        continue # 丢弃并检查下一个
                    else: # item_frame_id > local_frame_id
                        future_items.append(item) # 这是一个未来的帧，暂存
                except queue.Empty:
                    break
            
            # 把所有暂存的未来帧放回队列
            for item in future_items:
                self._output_queue.put(item)

            if output_item:
                # 成功拿到了匹配的说话帧
                frame = output_item['frame']
                speech_id = output_item['speech_id']
                avatar_status = AvatarStatus.SPEAKING
                end_of_speech = output_item['end_of_speech']
                audio_segment = output_item['audio_segment']
                
                # [新增] 更新延迟测量
                with self._frame_id_lock:
                    req_time = self._frame_generation_timestamps.pop(local_frame_id, None)
                if req_time:
                    latency = (frame_delivery_time - req_time) * 1000  # ms
                    self._latency_measurements.append(latency)
                    self._pipeline_latency_ms = sum(self._latency_measurements) / len(self._latency_measurements)
                    if self._config.debug:
                        logger.info(f"Frame {local_frame_id} delivered. Latency: {latency:.1f}ms. New Avg Latency: {self._pipeline_latency_ms:.1f}ms")

            else:
                # 没找到匹配的帧，播放静态帧作为填充
                frame = self._avatar.generate_idle_frame(local_frame_id)
                speech_id = last_active_speech_id
                avatar_status = AvatarStatus.LISTENING
                end_of_speech = False
                audio_segment = None

            # --- 后续的通知和状态管理逻辑保持不变 ---
            is_speaking = (avatar_status == AvatarStatus.SPEAKING)
            
            # Notify video
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            self._notify_video(VideoResult(video_frame=video_frame, speech_id=speech_id, avatar_status=avatar_status, end_of_speech=end_of_speech))

            # Audio related
            if audio_segment is not None and len(audio_segment) > 0:
                self._notify_audio(AudioResult(audio_frame=av.AudioFrame.from_ndarray(audio_segment[np.newaxis, :], format="flt", layout="mono"), speech_id=speech_id, end_of_speech=end_of_speech))
                if speech_id not in self._audio_cache: self._audio_cache[speech_id] = []
                self._audio_cache[speech_id].append(audio_segment)

            # 日志和状态切换
            if is_speaking and speech_id != current_speech_id:
                logger.info(f"[SPEAKING_FRAME] Start: frame_id={local_frame_id}, speech_id={speech_id}")
                current_speech_id = speech_id
            if end_of_speech:
                logger.info(f"Status change: SPEAKING -> LISTENING, frame_id={local_frame_id}, speech_id={speech_id}")
                self._notify_status_change(speech_id, AvatarStatus.LISTENING)
                if speech_id in self._audio_cache: del self._audio_cache[speech_id]
                current_speech_id = None
                
            local_frame_id += 1

    def _notify_audio(self, audio_result: AudioResult):
        if self.audio_output_queue:
            try:
                audio_data = audio_result.audio_frame.to_ndarray()
                self.audio_output_queue.put_nowait(audio_data)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_audio: {e}")

    def _notify_video(self, video_result: VideoResult):
        if self.video_output_queue:
            try:
                data = video_result.video_frame.to_ndarray(format="bgr24")
                self.video_output_queue.put_nowait(data)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_video: {e}")

    def _notify_status_change(self, speech_id: str, status: AvatarStatus):
        if self.event_out_queue and status == AvatarStatus.LISTENING:
            try:
                from handlers.avatar.liteavatar.avatar_handler_liteavatar import Tts2FaceEvent
                self.event_out_queue.put_nowait(Tts2FaceEvent.SPEAKING_TO_LISTENING)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_status_change: {e}")

    def _clear_queues(self):
        # [修改] 移除了 _frame_id_queue
        queues_to_clear = [self._audio_queue, self._whisper_queue, self._unet_queue, self._compose_queue, self._output_queue]
        for q in queues_to_clear:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass