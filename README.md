# Video Summarizer CLI

基于 AI 的视频内容总结命令行工具。输入一段视频，自动提取语音转录和关键帧画面，通过大模型生成结构化的 Markdown 总结文档。

## 工作原理

```
                    ┌─────────────────────────────────────┐
                    │           视频文件 (MP4/...)           │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           ┌──────────────┐              ┌──────────────┐
           │  音频提取      │              │  关键帧提取    │
           │  (FFmpeg)     │              │  (FFmpeg)     │
           └──────┬───────┘              └──────┬───────┘
                  ▼                              ▼
           ┌──────────────┐              ┌──────────────┐
           │  语音识别      │              │  画面去重      │
           │  (ASR)       │              │  (感知哈希)    │
           └──────┬───────┘              └──────┬───────┘
                  │                              ▼
                  │                      ┌──────────────┐
                  │                      │  视觉分析      │
                  │                      │  (智谱 Vision) │
                  │                      └──────┬───────┘
                  │                              │
                  └──────────┬───────────────────┘
                             ▼
                      ┌──────────────┐
                      │  AI 综合总结   │
                      │  (智谱 GLM)   │
                      └──────┬───────┘
                             ▼
                      ┌──────────────┐
                      │  summary.md  │
                      └──────────────┘
```

处理流程分为 5 个阶段：

1. **音频提取** — 通过 FFmpeg 从视频中提取 16kHz 单声道 WAV 音频
2. **语音识别 (ASR)** — 使用 faster-whisper / mlx-whisper 等引擎将音频转为带时间戳的文字转录
3. **关键帧提取** — 通过 FFmpeg 提取 I 帧，再用感知哈希 (pHash) 去除相似画面
4. **视觉分析** — 将关键帧发送至智谱视觉大模型，分类画面类型 (IDE/终端/PPT 等) 并提取文字和语义描述
5. **AI 总结** — 将转录文本和视觉分析结果合并，由智谱 GLM 大模型生成结构化 Markdown 总结

其中，阶段 1-2（音频分支）与阶段 3-4（视觉分支）**并行执行**，阶段 5 等待两个分支全部完成后执行。

## 功能特性

- **多 ASR 引擎支持** — faster-whisper (CUDA/CPU)、mlx-whisper (Apple Silicon)、Qwen3-ASR
- **自动硬件适配** — 检测 GPU 类型，自动选择最优引擎和计算精度
- **智能关键帧去重** — 感知哈希过滤相似画面，避免冗余分析
- **并发视觉分析** — 多线程并发调用视觉 API，可配置并发数
- **断点续跑** — `--resume` 跳过已完成的阶段，从中间产物恢复
- **单阶段执行** — `--only` 指定只运行某个阶段，方便调试
- **灵活配置** — 5 层配置优先级：代码默认值 < YAML 配置 < 环境覆盖 < CLI 参数 < 环境变量
- **交互模式** — 引导式输入 API Key 和视频路径，自动保存到 `.env`

## 实验环境

| 设备 | CPU/GPU | 内存/显存 | 角色 |
|------|---------|----------|------|
| Macbook Pro M4 | Apple M4 芯片 | 48GB 统一内存 | 开发机 + 备用执行机 |
| Windows PC | RTX 4070Ti (12GB VRAM) | 128GB RAM | 主要 GPU 执行机 |

两台机器均可用，系统自动检测硬件环境并选择最优执行策略。

## 环境要求

- **Python** >= 3.11
- **FFmpeg** — 需安装并加入系统 PATH
- **智谱 AI API Key** — 用于视觉分析和 AI 总结 ([获取地址](https://open.bigmodel.cn/))
- **CUDA GPU** (可选) — 加速 ASR 转录；无 GPU 时自动回退 CPU 模式

## 安装

```bash
git clone <repository-url>
cd video-summarizer-cli

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

确保 FFmpeg 已安装：

```bash
ffmpeg -version
```

## 快速开始

### 设置 API Key

方式一：环境变量

```bash
export ZAI_API_KEY="your-api-key-here"
```

方式二：交互模式自动引导

```bash
python -m src.main summarize -i
```

首次运行交互模式会提示输入 API Key 并询问是否保存到 `.env` 文件。

### 基本用法

```bash
# 总结视频
python -m src.main summarize /path/to/video.mp4

# 交互模式（引导输入）
python -m src.main summarize -i

# 指定输出目录
python -m src.main summarize /path/to/video.mp4 -o ./my-output

# 恢复中断的任务
python -m src.main summarize /path/to/video.mp4 --resume

# 只运行某个阶段
python -m src.main summarize /path/to/video.mp4 --only asr
python -m src.main summarize /path/to/video.mp4 --only visual
python -m src.main summarize /path/to/video.mp4 --only summary
```

### 查看环境信息

```bash
python -m src.main env-info
```

输出示例：

```
┌─────────────────── Environment Information ───────────────────┐
│ Property         │ Value                                     │
├──────────────────┼───────────────────────────────────────────┤
│ Environment      │ mac                                       │
│ OS               │ darwin                                    │
│ CUDA Available   │ False                                     │
│ MPS Available    │ True                                      │
│ GPU              │ Apple M2 Pro                              │
│ CPU Cores        │ 12                                       │
│ RAM              │ 16.0 GB                                   │
└──────────────────┴───────────────────────────────────────────┘
```

## CLI 参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `VIDEO_PATH` | — | 视频文件路径（非交互模式下必填） |
| `--interactive` | `-i` | 启用交互模式 |
| `--output DIR` | `-o` | 输出目录 |
| `--asr-engine` | — | ASR 引擎：`faster_whisper` / `qwen3_asr` / `mlx_whisper` |
| `--visual-model` | — | 视觉模型名称 |
| `--summary-model` | — | 总结模型名称 |
| `--config` | — | 配置文件路径 |
| `--only` | — | 只运行指定阶段：`audio` / `asr` / `keyframe` / `visual` / `summary` |
| `--resume` | — | 从已有中间产物恢复执行 |

## 配置

配置文件位于 `config/config.yml`，支持以下配置项：

### ASR 语音识别

```yaml
asr:
  engine: faster_whisper  # faster_whisper / qwen3_asr / mlx_whisper
  engines:
    faster_whisper:
      model: large-v3-turbo    # 模型大小：small / medium / large-v3 / large-v3-turbo
      compute_type: float16    # 计算精度：float16 / int8
      device: auto             # auto / cuda / cpu
      language: null           # null = 自动检测
```

### 关键帧提取

```yaml
keyframe:
  method: keyframe           # keyframe (I帧) / interval (定时)
  format: png
  dedup_threshold: 0.95      # 去重相似度阈值 (0-1)
  max_frames: 500            # 最大提取帧数
```

### 视觉分析

```yaml
visual:
  model: glm-4.6v-flashx    # glm-4.6v-flashx / glm-4.6v-flash / glm-5v-turbo / glm-4.5v
  concurrency: 5             # 并发请求数
  timeout: 30                # 超时秒数
  retry: 3                   # 失败重试次数
```

### AI 总结

```yaml
summary:
  model: glm-4.7             # glm-5.1 / glm-5 / glm-4.7 / glm-4.6 / glm-4.5-air
  max_tokens: 8000
  timeout: 120
  retry: 3
```

### 输出

```yaml
output:
  dir: null                  # null = ./output/
  include_screenshots: true   # Markdown 中嵌入截图引用
  keep_intermediate: true    # 保留中间文件 (audio.wav, transcript.json 等)
```

### 配置优先级

从低到高：

1. **代码默认值** — `config_loader.py` 中的 `DEFAULT_CONFIG`
2. **YAML 配置文件** — `config/config.yml`
3. **环境覆盖** — `env_overrides` 根据运行环境 (mac / windows_gpu) 自动应用
4. **CLI 参数** — `--asr-engine`、`--visual-model` 等
5. **环境变量** — `ZAI_API_KEY`

## 硬件环境路由

| 环境 | ASR 引擎 | 设备 |
|------|---------|------|
| Windows + NVIDIA GPU | faster-whisper | CUDA |
| Mac (Apple Silicon) | mlx-whisper | Metal/MLX |
| 无 GPU | faster-whisper | CPU |

通过 `env_detector.py` 自动检测硬件，结合 `config.yml` 中的 `env_overrides` 自动路由。

## 输出结构

运行后在输出目录下生成以下文件：

```
output/
└── <video_name>/
    ├── audio.wav                # 提取的音频
    ├── transcript.json          # ASR 转录结果
    ├── keyframes/               # 关键帧图片
    │   ├── key_000001.png
    │   ├── key_000042.png
    │   └── ...
    ├── visual_analysis.json     # 视觉分析结果
    └── summary.md               # 最终总结文档
```

## 项目结构

```
src/
├── main.py           # CLI 入口 (click)：summarize、env-info 命令
├── interactive.py    # 交互模式：API Key 设置、视频路径输入
├── pipeline.py       # 流水线编排：音频+ASR ∥ 关键帧+视觉 → 总结
├── config_loader.py  # 5 层配置加载与合并
├── env_detector.py   # 硬件环境检测 (CUDA / MPS / CPU)
├── context.py        # RuntimeContext 数据类 + 6 种自定义异常
├── stages/           # 流水线各阶段实现
│   ├── audio_extractor.py   # 音频提取
│   ├── asr.py               # 语音识别
│   ├── keyframe.py          # 关键帧提取
│   ├── visual.py            # 视觉分析
│   └── summarizer.py        # AI 总结生成
└── utils/            # 外部服务封装
    ├── ffmpeg.py             # FFmpeg 子进程封装
    ├── zhipu_client.py       # 智谱 AI API 客户端 (含重试)
    ├── image_hash.py         # 感知哈希去重
    └── logger.py             # 日志配置
config/
└── config.yml        # 运行时配置
tests/                # 测试用例 (与 src/ 镜像对应)
```

## 开发

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 设置 API Key（任选一种）
export ZAI_API_KEY="your-api-key"   # 方式一：环境变量
# 或使用交互模式运行时自动引导输入          # 方式二：交互引导

# 运行项目（未配置 entrypoint，通过 -m 方式运行）
python -m src.main summarize /path/to/video.mp4
python -m src.main summarize -i                    # 交互模式
python -m src.main env-info                         # 查看环境信息

# 运行测试
pytest tests/ -v

# 单模块测试
pytest tests/stages/test_asr.py -v

# 类型检查
mypy src/
```

## License

MIT
