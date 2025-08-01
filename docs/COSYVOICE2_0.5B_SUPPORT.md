# CosyVoice2-0.5B Model Support

This document explains how to configure OpenAvatarChat to use CosyVoice2-0.5B local model.

## Overview

OpenAvatarChat now supports explicit model type selection for CosyVoice models, including the CosyVoice2-0.5B variant. This enhancement allows users to:

1. Explicitly specify which model type to use (CosyVoice or CosyVoice2)
2. Use automatic model type detection
3. Better support for different model variants including CosyVoice2-0.5B

## Configuration Options

### Model Type Selection

The `model_type` field in the CosyVoice configuration supports three values:

- `"auto"` (default): Automatically detects the model type by trying CosyVoice first, then CosyVoice2
- `"cosyvoice"`: Explicitly use the original CosyVoice model loader
- `"cosyvoice2"`: Explicitly use the CosyVoice2 model loader (recommended for 0.5B models)

### Configuration Example for CosyVoice2-0.5B

```yaml
CosyVoice:
  enabled: True
  module: tts/cosyvoice/tts_handler_cosyvoice
  model_name: "iic/CosyVoice2-0.5B"  # or path to your local model
  model_type: "cosyvoice2"  # Explicitly specify CosyVoice2
  spk_id: "中文女"  # Speaker ID for SFT models
  sample_rate: 24000
  process_num: 2
```

### Zero-shot Configuration

For zero-shot inference with reference audio:

```yaml
CosyVoice:
  enabled: True
  module: tts/cosyvoice/tts_handler_cosyvoice
  model_name: "iic/CosyVoice2-0.5B"
  model_type: "cosyvoice2"
  ref_audio_path: "path/to/your/reference/audio.wav"
  ref_audio_text: "Reference audio transcription text"
  sample_rate: 24000
  process_num: 2
```

## Usage

1. **Use the provided configuration**: Use `config/chat_with_cosyvoice2_0.5b.yaml` as a starting point
2. **Update model path**: Change `model_name` to point to your CosyVoice2-0.5B model location
3. **Configure speaker**: Set appropriate `spk_id` for SFT models or `ref_audio_path`/`ref_audio_text` for zero-shot
4. **Start the application**: Run with your configuration file

```bash
python src/demo.py --config config/chat_with_cosyvoice2_0.5b.yaml
```

## Troubleshooting

### Model Loading Issues

If you encounter model loading issues:

1. **Check model type**: Ensure `model_type` is set to `"cosyvoice2"` for 0.5B models
2. **Verify model path**: Ensure the model is properly downloaded and accessible
3. **Check logs**: The enhanced logging will show which model loader is being used
4. **Try auto mode**: Set `model_type: "auto"` to let the system detect the correct loader

### Error Messages

- `"No valid model_type! Both CosyVoice and CosyVoice2 failed to load."`: The model cannot be loaded with either loader. Check model format and path.
- Model-specific loading errors will now be logged separately for better debugging.

## Model Download

For ModelScope models like `iic/CosyVoice2-0.5B`, the system will automatically download them if not already present. For local models, ensure they are properly placed in the models directory or provide the full path.