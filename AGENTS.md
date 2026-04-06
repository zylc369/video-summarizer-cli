# video-summarizer-cli — 项目知识库

## 项目概述

CLI 工具：读取视频 → 提取音频+关键帧 → ASR 转写 + 视觉分析 → AI 综合总结 → 输出结构化 Markdown。
技术栈：Python 3.11+，智谱 API (zhipuai SDK)，FFmpeg，faster-whisper / Qwen3-ASR。

## 目录约定

```
video-summarizer-cli/
├── src/                  # 源代码（Git 根目录）
│   ├── main.py           # CLI 入口 (click)
│   ├── context.py        # 全局 RuntimeContext
│   ├── pipeline.py       # 管道编排
│   ├── config_loader.py  # YAML 配置加载与合并
│   ├── env_detector.py   # 运行环境检测 (mac / windows_gpu)
│   ├── stages/           # 管道各阶段实现
│   │   ├── audio_extractor.py
│   │   ├── asr.py
│   │   ├── keyframe.py
│   │   ├── visual.py
│   │   └── summarizer.py
│   └── utils/            # 工具函数
│       ├── ffmpeg.py
│       ├── zhipu_client.py
│       ├── image_hash.py
│       └── logger.py
├── config/
│   └── config.yml        # 全局 YAML 配置
├── docs/                 # 文档目录
│   └── 需求/             # 需求文档
├── tests/                # 测试（与 src/ 平级）
├── output/               # 运行时输出（gitignore）
└── requirements.txt
```

---

## Python 编写规范

### 一、语言版本与类型系统

- **Python 3.11+**，使用现代语法（`X | Y` 联合类型、`match` 语句等）
- **强制类型注解**：所有函数签名必须标注参数和返回值类型
- 使用 `dataclass` 定义数据结构，不用 TypedDict（除非需要 JSON 互操作）
- 禁止 `# type: ignore`、`# type: bypass`、`cast(Any, ...)` — 必须从根源修复类型问题
- 复杂类型使用 `TypedDict` 或 `dataclass`，不要用裸 `dict[str, Any]`

```python
# ✅ 正确
def extract_audio(video_path: Path, output_dir: Path) -> Path:
    ...

# ❌ 错误
def extract_audio(video_path, output_dir):
    ...
```

### 二、代码风格

- 遵循 **PEP 8**，行宽上限 **120 字符**
- 缩进：4 空格，禁止 Tab
- 字符串：优先双引号 `"`，f-string 用于格式化
- 命名规范：

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块/包 | snake_case | `audio_extractor.py` |
| 函数/方法 | snake_case | `detect_environment()` |
| 类 | PascalCase | `RuntimeContext` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_ASR_ENGINE` |
| 私有属性/方法 | _前缀 | `_merge_env_overrides()` |
| 配置键 | snake_case | `compute_type` |

### 三、导入规范

```python
# 导入顺序（每组之间空一行）：
# 1. 标准库
import os
import platform
from pathlib import Path
from dataclasses import dataclass

# 2. 第三方库
import click
import torch
from rich.console import Console

# 3. 本项目模块
from src.context import RuntimeContext
from src.utils.ffmpeg import run_ffmpeg
```

- 禁止 `from module import *`（通配符导入）
- 禁止相对导入中使用超过一级：`from ..utils import x` ✗，`from .utils import x` ✓
- 第三方库导入按字母序排列

### 四、函数设计

- 单一职责：一个函数只做一件事
- 函数体不超过 **50 行**，超过则拆分
- 参数不超过 **5 个**，超过则使用 `dataclass` 或 `TypedDict` 封装
- 纯函数优先：避免隐式依赖全局状态（`RuntimeContext` 通过参数显式传递）

```python
# ✅ 正确：显式传递 context
def run_asr(context: RuntimeContext, audio_path: Path) -> list[TranscriptSegment]:
    engine = resolve_asr_engine(context)
    ...

# ❌ 错误：隐式依赖全局变量
_context: RuntimeContext | None = None

def run_asr(audio_path: Path) -> list[dict]:
    engine = _context.config["asr"]["engine"]  # 隐式全局
    ...
```

### 五、错误处理

- **禁止空 `except` 块**，禁止 `except Exception: pass`
- 捕获具体异常，不要裸 `except:` 或 `except Exception`
- 异常必须执行有效操作：向上抛出 / 记录完整堆栈 / 返回明确错误信息
- 使用自定义异常类区分错误类型（不要用 `raise ValueError("...")` 处理所有错误）

```python
# ✅ 正确
class FFmpegNotFoundError(Exception): ...
class ASREngineError(Exception): ...

def extract_audio(video_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise FFmpegNotFoundError("FFmpeg not found. Install: https://ffmpeg.org")
    try:
        run_ffmpeg(...)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Audio extraction failed: {e.stderr}") from e

# ❌ 错误
try:
    run_ffmpeg(...)
except Exception:
    pass  # 吞没异常
```

### 六、配置管理

- **禁止在代码中硬编码**模型名称、API 地址、硬件参数等可变配置
- 所有可变参数通过 `config/config.yml` 管理
- 配置加载优先级：代码默认值 < YAML 文件 < 环境变量 < CLI 参数
- 环境特定配置通过 `env_overrides` 机制自动合并

```python
# ✅ 正确：从配置读取
model = context.config["asr"]["engines"]["faster_whisper"]["model"]

# ❌ 错误：硬编码
model = "large-v3-turbo"
```

### 七、日志规范

- 使用标准 `logging` 模块，通过 `src/utils/logger.py` 统一配置
- 日志级别使用规范：
  - `DEBUG`：详细调试信息（帧数、文件路径、配置值）
  - `INFO`：关键流程节点（阶段开始/完成、环境检测结果）
  - `WARNING`：可恢复的异常（CUDA 不可用回退 CPU、API 重试）
  - `ERROR`：不可恢复错误（文件无法读取、API 重试耗尽）
- 使用 `rich` 库美化 CLI 输出（进度条、状态表格），不用于日志

```python
logger = logging.getLogger(__name__)

# ✅ 正确
logger.info("ASR transcription completed: %d segments, %.1fs", len(segments), duration)

# ❌ 错误
print("ASR done")  # 不要用 print 输出运行信息
```

### 八、并发与异步

- 管道阶段 1-4（音频提取+ASR 与 关键帧提取+视觉分析）**并行执行**
- 使用 `concurrent.futures.ThreadPoolExecutor` 或 `asyncio`，不要用裸线程
- API 调用（智谱视觉分析）使用信号量控制并发数（`concurrency` 配置项）
- 禁止无限制的并发，必须设置上限

### 九、测试规范

- 测试框架：**pytest**
- 测试目录结构与源码镜像：`tests/stages/test_asr.py` 对应 `src/stages/asr.py`
- 测试命名：`test_<函数名>_<场景>_<预期结果>`
- Fixture 放在 `tests/conftest.py`，按需拆分
- **禁止修改测试使测试通过**：测试失败必须修复业务代码
- 使用 `pytest.mark.parametrize` 覆盖多种输入
- Mock 外部依赖（FFmpeg、API 调用、GPU 检测），不 Mock 被测单元的内部逻辑

```python
# ✅ 测试命名示例
def test_detect_environment_mac_with_mps(mock_no_cuda, mock_mps):
    ...

def test_extract_audio_missing_ffmpeg_raises(tmp_path):
    ...
```

### 十、文件与路径处理

- 统一使用 `pathlib.Path`，禁止 `os.path.join`、`os.path.exists` 等 `os.path` 系列调用
- 输出目录结构由 `RuntimeContext.output_dir` 统一管理
- 临时文件使用 `tempfile` 或 `RuntimeContext.temp_dir`，确保清理
- 路径拼接用 `/` 运算符：`output_dir / video_name / "audio.wav"`

```python
# ✅ 正确
output_path = context.output_dir / video_stem / "audio.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)

# ❌ 错误
output_path = os.path.join(context.output_dir, video_stem, "audio.wav")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

### 十一、外部调用封装

- FFmpeg 调用统一通过 `src/utils/ffmpeg.py` 封装，禁止各阶段直接 `subprocess.run`
- 智谱 API 调用统一通过 `src/utils/zhipu_client.py` 封装，包含重试、超时、错误处理
- 封装层负责：参数校验、错误转换、日志记录、资源清理

### 十二、数据结构定义

- ASR 转录结果、视觉分析结果、配置节等均使用 `dataclass` 定义
- JSON 序列化/反序列化通过 dataclass 方法，不直接操作裸 `dict`
- 不要在不同阶段之间传递原始字典，使用类型化的 dataclass

```python
@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    language: str | None = None

@dataclass
class VisualAnalysisResult:
    frame_path: Path
    timestamp: float
    frame_type: str  # "ide" | "terminal" | "ppt" | "ida_pro" | "other"
    text_content: str
    description: str
```

---

## 运行命令

```bash
# 开发环境设置
python -m venv .venv
source .venv/bin/activate  # macOS
pip install -r requirements.txt

# 运行
python -m src.main summarize /path/to/video.mp4
python -m src.main env-info

# 测试
pytest tests/ -v
pytest tests/stages/test_asr.py -v  # 单个模块

# 类型检查（如有配置）
mypy src/
```

## 硬件环境路由

| 环境 | ASR 引擎 | 设备 |
|------|---------|------|
| Windows + CUDA GPU | faster-whisper | CUDA |
| Mac (Apple Silicon) | mlx-whisper | MPS/MLX |
| 无 GPU | faster-whisper | CPU |

## 反模式（禁止）

- `# type: ignore`、`cast(Any, x)` — 修复类型，不要压制
- `except Exception: pass` — 不吞异常
- `print()` 用于运行信息 — 用 `logging`
- `os.path.*` — 用 `pathlib.Path`
- 硬编码模型名称 / API 地址 / 硬件参数 — 用配置
- 裸 `subprocess.run` 调用 FFmpeg — 通过 `utils/ffmpeg.py`
- 函数参数 > 5 个 — 用 dataclass 封装
- 测试失败时修改测试代码 — 修复业务代码
