import os
import sys
import unittest
import numpy as np
import requests
import torchaudio


class TestCosyVoice(unittest.TestCase):
    def test1(self):
        pass


if __name__ == '__main__':

    # url = "http://127.0.0.1:5000/inference_zero_shot"
    # payload = {
    #     'tts_text': '说句话呀,你这样显得这个文字有点短呀，要怎么样才能说的长一点呢',
    # }
    # print(payload)
    # response = requests.request('GET', url, data=payload, stream=True)
    # print(response)
    # tts_audio = b''
    # for r in response.iter_content(chunk_size=16000):
    #     tts_audio += r
    # tts_speech = np.array(np.frombuffer(tts_audio, dtype=np.int16))
    sys.path.append(os.getcwd())
    from engine_utils.directory_info import DirectoryInfo

    sys.path.append(os.path.join(DirectoryInfo.get_src_dir(), 'third_party', 'CosyVoice'))
    sys.path.append(os.path.join(DirectoryInfo.get_src_dir(), 'third_party',
                    'CosyVoice', 'third_party', 'Matcha-TTS'))
    from src.third_party.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    model = None
    model_name = 'iic/CosyVoice-300M-SFT'
    model_type = 'auto'  # Can be 'auto', 'cosyvoice', 'cosyvoice2'
    
    # Test model loading with different types
    if model_type.lower() == "cosyvoice":
        print(f"Loading CosyVoice model: {model_name}")
        model = CosyVoice(model_dir=model_name)
    elif model_type.lower() == "cosyvoice2":
        print(f"Loading CosyVoice2 model: {model_name}")
        model = CosyVoice2(model_dir=model_name)
    else:  # auto mode
        print(f"Auto-detecting model type for: {model_name}")
        try:
            print("Trying CosyVoice first...")
            model = CosyVoice(model_dir=model_name)
            print("Successfully loaded with CosyVoice")
        except Exception as e:
            print(f"CosyVoice failed: {e}, trying CosyVoice2...")
            try:
                model = CosyVoice2(model_dir=model_name)
                print("Successfully loaded with CosyVoice2")
            except Exception as e2:
                print(f"Both failed. CosyVoice error: {e}, CosyVoice2 error: {e2}")
                raise TypeError('No valid model_type! Both CosyVoice and CosyVoice2 failed to load.')
    res = model.inference_sft('说句话呀，白色的咖啡杯放在桌子上。白色的咖啡杯放在桌子上。', '中文女')
    print(res)
    for item in res:
        print(item)
