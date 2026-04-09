# src/utils — Cross-Cutting Utility Wrappers

## OVERVIEW
Authorized gateways for external services: FFmpeg, Zhipu AI API, logging, image hash.

## WHERE TO LOOK
| File | Purpose | Usage |
|------|---------|-------|
| ffmpeg.py | FFmpeg subprocess calls | Audio extraction, frame extraction |
| zhipu_client.py | Zhipu AI API client | Vision analysis, chat completion |
| image_hash.py | Perceptual hash for deduplication | Keyframe filtering |
| logger.py | Logging configuration | All modules |

## CONVENTIONS

### FFmpeg Wrapper
- ALL FFmpeg calls must use `run_ffmpeg(args, timeout) -> CompletedProcess`
- `check_ffmpeg_available()` first or catch `FFmpegNotFoundError`
- Args passed as list: `["-i", str(video_path), "-vn", "-acodec", "pcm_s16le", str(output)]`

### Zhipu API Client
- Factory: `create_zhipu_client(api_key) -> ZhipuClient`
- Methods: `chat_completion()`, `vision_analysis()` (both retryable via `_call_with_retry()`)
- Retryable: rate limits, timeouts. Non-retryable: auth, invalid params. See `_RETRYABLE_ERRORS` constant

### Image Hash
- `compute_phash()` → ImageHash, `are_similar(hash1, hash2, threshold) -> bool`
- `deduplicate_frames()` filters near-duplicates, returns unique frame paths

### Logger
- `setup_logging(level=INFO, fmt=...)` configures root logger once (usually in main.py)
- `get_logger(name)` for per-module loggers

## ANTI-PATTERNS
- Direct `subprocess.run` for FFmpeg → MUST use `run_ffmpeg()`
- Direct `zhipuai` SDK calls → MUST use `ZhipuClient`
- Bypassing retry logic → Use `_call_with_retry()` wrapper methods
- Perceptual hash thresholds hardcoded → Pass via `threshold` param
- Multiple logging configs → Call `setup_logging()` once at entry point
