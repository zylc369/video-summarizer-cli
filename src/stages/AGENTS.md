# src/stages/ — Pipeline Stage Implementations

## OVERVIEW
Pipeline stage modules: audio extraction, ASR, keyframe extraction, visual analysis, and summary generation.

## WHERE TO LOOK

| File | Core Functions | Returns |
|------|----------------|---------|
| `audio_extractor.py` | `extract_audio()` | `Path` (audio file) |
| `asr.py` | `run_asr()`, `resolve_asr_engine()` | `list[TranscriptSegment]`, `str` |
| `keyframe.py` | `extract_keyframes()`, `parse_frame_timestamp()` | `list[Path]`, `float | None` |
| `visual.py` | `analyze_frames()` | `list[VisualAnalysisResult]` |
| `summarizer.py` | `generate_summary()` | `Path` (markdown file) |

## CONVENTIONS

- Context Passing: All public functions accept `context: RuntimeContext` as first parameter. Never global.
- Typed Returns: Return typed dataclasses (`TranscriptSegment`, `VisualAnalysisResult`). Never raw dicts.
- Error Handling: Raise custom exceptions from `src/context.py`: `FFmpegError`, `ASREngineError`, `VisualAnalysisError`, `SummarizerError`.
- External Dependencies: API calls through utils wrappers (`utils/zhipu_client.py`, `utils/ffmpeg.py`). Never direct external clients.
- Concurrency: Visual analysis uses `ThreadPoolExecutor` with semaphore from config. ASR engine routed via `resolve_asr_engine()`.
- Logging: `logging.getLogger(__name__)`. Never `print()`.
- Parallel Orchestration: Audio+ASR branch runs concurrently with keyframe+visual branch. Orchestrated by `pipeline.py`.

## ANTI-PATTERNS

- Global context access or implicit dependencies
- Returning raw dicts instead of typed dataclasses
- Direct external API calls (FFmpeg subprocess, Zhipu API) — use utils wrappers
- Raising generic exceptions (`ValueError`, `RuntimeError`) — use custom exceptions
- `print()` statements for status/progress — use logging
- Hardcoding model names or API parameters — read from config
