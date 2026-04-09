# video-summarizer-cli

**Generated:** 2026-04-08 | **Commit:** cf0eb44 | **Branch:** main

## OVERVIEW

CLI: video → audio + keyframes → ASR + visual analysis → AI summary → Markdown.
Stack: Python 3.11+, zhipuai SDK, FFmpeg, faster-whisper / Qwen3-ASR / mlx-whisper.

## STRUCTURE

```
src/
├── main.py           # CLI entry (click): summarize, env-info
├── interactive.py    # Interactive mode: API key setup, video path prompt
├── pipeline.py       # Parallel orchestration: audio+ASR ∥ keyframe+visual → summary
├── config_loader.py  # 5-layer config merge: defaults < YAML < env_overrides < CLI < env vars
├── env_detector.py   # Hardware detection: CUDA / MPS / CPU routing
├── context.py        # RuntimeContext dataclass + 6 custom exceptions
├── stages/           # Pipeline stages (see src/stages/AGENTS.md)
└── utils/            # External call wrappers (see src/utils/AGENTS.md)
config/config.yml     # Runtime config (ASR engines, models, concurrency, output)
tests/                # Mirrors src/ structure
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add CLI command | `src/main.py` | Register under `@cli` click group |
| Add pipeline stage | `src/stages/` | New module, called from `pipeline.py` |
| Change ASR/visual model | `config/config.yml` | Never hardcode — read from config |
| Add external API wrapper | `src/utils/` | Must include retry, timeout, error conversion |
| Change config priority | `src/config_loader.py` | `load_config()` orchestrates 5-layer merge |
| Hardware detection | `src/env_detector.py` | `detect_system_info()` → env_name, GPU info |
| API key management | `src/interactive.py` | `.env` file, env var, or prompt |
| Test a module | `tests/` | Mirror path: `tests/stages/test_asr.py` ↔ `src/stages/asr.py` |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `RuntimeContext` | dataclass | `context.py:31` | 5 | Shared pipeline state |
| `load_config` | function | `config_loader.py:197` | 1 | Config loading entry point |
| `run_pipeline` | function | `pipeline.py:20` | 1 | Parallel stage orchestration |
| `ZhipuClient` | class | `utils/zhipu_client.py:29` | 21 | AI API client (chat + vision) |
| `run_ffmpeg` | function | `utils/ffmpeg.py:30` | 11 | FFmpeg subprocess wrapper |
| `detect_system_info` | function | `env_detector.py:96` | 2 | Hardware + OS detection |
| `TranscriptSegment` | dataclass | `stages/asr.py:15` | 5 | ASR output type |
| `VisualAnalysisResult` | dataclass | `stages/visual.py:17` | 5 | Vision output type |

## CONVENTIONS

- **Python 3.11+**, modern syntax (`X | Y` unions, `match`), PEP 8, **120 char** line width
- **Strict typing**: `disallow_untyped_defs = true` (pyproject.toml). All signatures annotated.
- **dataclass for data**, not bare `dict[str, Any]`. Stages pass typed dataclasses, not raw dicts.
- **Context via parameter**: `RuntimeContext` always explicit first param. Never global/module-level state.
- **Config over hardcode**: model names, API params, hardware settings → `config/config.yml`. Priority: defaults < YAML < env_overrides < CLI < env vars.
- **External calls through utils**: FFmpeg → `utils/ffmpeg.py`, Zhipu API → `utils/zhipu_client.py`. Never direct `subprocess.run` or raw SDK calls in stages.
- **Custom exceptions**: `FFmpegNotFoundError`, `FFmpegError`, `ASREngineError`, `VisualAnalysisError`, `SummarizerError`, `ConfigError` — all in `context.py`. Never bare `ValueError` for domain errors.
- **Logging**: `logging.getLogger(__name__)` with `%s` formatting. Never `print()` for runtime info. `rich` for CLI output only.
- **Paths**: `pathlib.Path` exclusively. `/` operator for joins. Never `os.path.*`.
- **Functions**: ≤50 lines, ≤5 params (else use dataclass). Single responsibility.
- **Concurrency**: `ThreadPoolExecutor`, never bare threads. Semaphore for API calls (`concurrency` config). Audio+ASR ∥ keyframe+visual.
- **Imports**: stdlib → third-party (alpha) → project. No wildcard. One-level relative only (`from .x import y` ok, `from ..x` forbidden).

## ANTI-PATTERNS (FORBIDDEN)

- `# type: ignore`, `cast(Any, x)` — fix the type, don't suppress
- `except Exception: pass` — never swallow exceptions
- `print()` for runtime info — use `logging`
- `os.path.*` — use `pathlib.Path`
- Hardcoded model names / API URLs / hardware params — use config
- Direct `subprocess.run` for FFmpeg — go through `utils/ffmpeg.py`
- Raw dicts between pipeline stages — use dataclasses
- `>5` function parameters — wrap in dataclass
- Modifying tests to make them pass — fix the business code

## HARDWARE ENVIRONMENT ROUTING

| Environment | ASR Engine | Device |
|-------------|-----------|--------|
| Windows + CUDA GPU | faster-whisper | CUDA |
| Mac (Apple Silicon) | mlx-whisper | MPS/MLX |
| No GPU | faster-whisper | CPU |

Detected by `env_detector.py`, routed via `config/config.yml` `env_overrides`.

## COMMANDS

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.main summarize /path/to/video.mp4
python -m src.main env-info

pytest tests/ -v
pytest tests/stages/test_asr.py -v  # single module
mypy src/
```

## NOTES

- No `[project.scripts]` in pyproject.toml — must use `python -m src.main`
- No `conftest.py` — fixtures defined per test file using `tmp_path`
- `src/__main__.py` uses absolute import (`from src.main import cli`)
- Known type issue: `_format_timestamp` in `summarizer.py` accepts `None` but annotation says `float` — fix to `float | None`
- `output/` directory is gitignored — runtime artifacts only
