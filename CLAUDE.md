# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenAvatarChat is a modular interactive digital human conversation implementation that can run full-featured on a single PC. The system supports multiple modes including MiniCPM-o as a multimodal language model or cloud-based APIs for ASR + LLM + TTS functionality.

## Key Commands

### Development Setup
- **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) 
- **Initialize submodules**: `git submodule update --init --recursive`
- **Install all dependencies**: `uv sync --all-packages`
- **Install config-specific dependencies**: `uv run install.py --uv --config <config-file>.yaml`
- **Post-installation setup**: `./scripts/post_config_install.sh --config <config-file>.yaml`

### Running the Application
- **Run with specific config**: `uv run src/demo.py --config <config-file>.yaml`
- **Docker deployment**: `./build_and_run.sh --config <config-file>.yaml`
- **Download models**: Use scripts in `scripts/` directory (e.g., `scripts/download_MiniCPM-o_2.6.sh`)

### Development Tools
- **Compile requirements**: `./scripts/compile_requirements.sh`
- **Create SSL certificates**: `scripts/create_ssl_certs.sh`
- **Setup TURN server**: `scripts/setup_coturn.sh`

## Architecture Overview

The system follows a modular handler-based architecture:

### Core Components
- **ChatEngine** (`src/chat_engine/`): Main orchestration engine that manages sessions and handlers
- **HandlerManager**: Dynamically loads and manages different handlers based on configuration
- **SessionContext**: Manages individual chat sessions with input/output queues
- **Configuration System**: YAML-based configuration with multiple preset modes

### Handler Types
The system is built around pluggable handlers organized by functionality:

- **Client Handlers** (`src/handlers/client/`): WebRTC and LAM rendering clients
- **VAD Handlers** (`src/handlers/vad/`): Voice Activity Detection (Silero VAD)
- **ASR Handlers** (`src/handlers/asr/`): Speech-to-text (SenseVoice)
- **LLM Handlers** (`src/handlers/llm/`): Language models (MiniCPM-o, OpenAI-compatible APIs)
- **TTS Handlers** (`src/handlers/tts/`): Text-to-speech (CosyVoice, Edge TTS, Bailian API)
- **Avatar Handlers** (`src/handlers/avatar/`): Digital humans (LiteAvatar, LAM, MuseTalk)

### Data Flow Architecture
The system processes real-time audio/video streams through a pipeline:
1. **Input**: WebRTC client captures audio/video
2. **VAD**: Detects speech activity
3. **ASR**: Converts speech to text
4. **LLM**: Generates response text
5. **TTS**: Converts text to speech
6. **Avatar**: Generates synchronized lip movements/expressions
7. **Output**: Streams back to WebRTC client

## Configuration System

### Preset Configurations (in `config/`)
- `chat_with_minicpm.yaml`: Local MiniCPM-o multimodal model
- `chat_with_gs.yaml`: LAM Gaussian Splatting with APIs
- `chat_with_openai_compatible.yaml`: API LLM + local CosyVoice
- `chat_with_openai_compatible_bailian_cosyvoice.yaml`: All API-based (lightest)
- `chat_with_openai_compatible_edge_tts.yaml`: API LLM + Edge TTS
- `chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml`: APIs + MuseTalk

### Key Configuration Patterns
- Each handler has an `enabled` flag and `module` path
- Model paths are relative to `models/` directory
- API keys can be set in config or environment variables (`.env` file)
- SSL certificates go in `ssl_certs/` directory

## Development Patterns

### Handler Development
- Each handler inherits from base classes in `src/chat_engine/common/`
- Handlers have their own `pyproject.toml` for modular dependencies
- Handler registration happens automatically via module discovery
- Use `HandlerContext` for accessing session data and communication

### Dependency Management
- Root `pyproject.toml` defines core dependencies and workspace structure
- Individual handler modules have separate `pyproject.toml` files
- Use `uv` for all dependency operations, not `pip`
- The `install.py` script intelligently installs only required handler dependencies

### Third-Party Integration
- Git submodules are used for external projects (in `src/third_party/`)
- Custom WebRTC component: `src/third_party/gradio_webrtc_videochat/`
- Model repositories are cloned into `models/` directory

## Testing and Debugging

### Common Issues
- Ensure CUDA version >= 12.4 for GPU acceleration
- For Windows CosyVoice: Use conda for pynini, then UV with `--active` flag
- Missing submodules: Re-run `git submodule update --init --recursive`
- RTX 50 series: Requires CUDA 12.8+ and corresponding PyTorch

### Performance Considerations
- Default response latency target: ~2.2 seconds
- CPU inference possible but slower (LiteAvatar can run 30fps on i9-13980HX)
- GPU memory requirements vary by configuration (20GB+ for unquantized MiniCPM-o)
- Use int4 quantized models for <10GB VRAM setups

## File Structure Notes

### Important Directories
- `src/handlers/`: All handler implementations organized by type
- `config/`: YAML configuration files for different deployment modes
- `models/`: Downloaded model files and weights
- `scripts/`: Installation and setup scripts
- `ssl_certs/`: SSL certificates for HTTPS/WebRTC
- `tests/`: Unit tests and integration tests

### Key Files
- `src/demo.py`: Main application entry point
- `install.py`: Smart dependency installer based on config
- `pyproject.toml`: Root project configuration with workspace setup
- `build_and_run.sh`: Docker deployment script

## Environment Variables

The system reads from `.env` file in project root:
- `DASHSCOPE_API_KEY`: For Bailian/Alibaba Cloud services
- `OPENAI_API_KEY`: For OpenAI-compatible APIs
- `MODELSCOPE_CACHE`: Model download cache location
- `VIRTUAL_ENV`: For UV environment detection
- `PYTHONUTF8=1`: For Windows encoding issues