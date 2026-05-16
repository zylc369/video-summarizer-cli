# GLM 视觉模型调用指南

## 概述

本项目通过 ZhipuAI SDK（`zai` 包）调用 GLM 系列视觉模型，对视频关键帧进行内容识别与分类。核心调用链路：

```
pipeline.py → visual.py → zhipu_client.py → ZhipuAI API
```

## 支持的模型

| 模型 | 说明 |
|------|------|
| `glm-4.6v-flashx` | 默认，速度快（推荐） |
| `glm-4.6v-flash` | 快速版本 |
| `glm-4.6v` | 标准版本 |
| `glm-5v-turbo` | 新一代快速版 |
| `glm-4.5v` | 上一代视觉模型 |

模型配置位于 `config/config.yml` 的 `visual.model` 字段。

## 调用流程

### 1. 图片读取与 Base64 编码

在 `src/stages/visual.py:188-190`，将图片文件读取并编码：

```python
image_data = frame_path.read_bytes()
base64_image = base64.b64encode(image_data).decode("utf-8")
```

注意：编码后是纯 base64 字符串，不含 `data:image/...` 前缀。前缀在 API 调用时拼接。

### 2. 构建多模odal消息

在 `src/utils/zhipu_client.py:103-114`，将文本 prompt 和图片组装为多模态消息：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
        ],
    }
]
```

关键点：
- 图片以 `image_url` 类型嵌入，值为 `data:image/png;base64,{base64字符串}` 格式的 data URI
- `content` 是一个列表，可同时包含文本和图片多个元素
- 遵循 OpenAI 兼容的多模态消息格式

### 3. 发送 API 请求

在 `src/utils/zhipu_client.py:116-128`，调用 ZhipuAI SDK：

```python
response = self._client.chat.completions.create(
    model=model,        # e.g. "glm-4.6v-flashx"
    messages=messages,  # 包含图片的多模态消息
    max_tokens=max_tokens,
    timeout=timeout,
)
content = response.choices[0].message.content
```

视觉分析与文本对话使用同一个 `chat.completions.create` 接口，区别仅在于 `messages` 中是否包含 `image_url` 类型的内容。

### 4. 并发控制

在 `src/stages/visual.py:51-55`，使用 `ThreadPoolExecutor` 并发分析多帧：

```python
with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = {
        executor.submit(_analyze_single_frame, client, model, fp, max_tokens, timeout): fp
        for fp in frame_paths
    }
```

并发数由 `config/config.yml` 的 `visual.concurrency` 控制（默认 5）。

### 5. 重试机制

在 `src/utils/zhipu_client.py:131-230`，实现指数退避重试：

- **可重试错误**：限流（`APIReachLimitError`）、服务端错误（`APIInternalError`）、超时（`APITimeoutError`）、连接失败（`APIRequestFailedError`）
- **不可重试错误**：认证失败（`APIAuthenticationError`）、4xx 客户端错误
- **退避策略**：`delay = 2^attempt * base_delay`（base_delay 默认 1 秒）
- **最大重试次数**：由 `visual.retry` 配置（默认 3 次）

## 配置项

所有配置在 `config/config.yml` 的 `visual` 节：

```yaml
visual:
  model: glm-4.6v-flashx   # 视觉模型
  api_key: null             # null = 从环境变量 ZAI_API_KEY 读取
  max_tokens: 2000          # 单帧最大输出 token
  concurrency: 5            # 并发请求数
  timeout: 30               # 单次请求超时（秒）
  retry: 3                  # 失败重试次数
```

## API Key 优先级

1. `config/config.yml` 中的 `visual.api_key`
2. 环境变量 `ZAI_API_KEY`
3. 交互模式下用户手动输入（存储在 `.env` 文件）

## Prompt 设计

在 `src/stages/visual.py:89-111`，Prompt 要求模型返回结构化文本：

```
TYPE: [ide/terminal/ppt/ida_pro/other]
TEXT: [提取的文本内容]
DESCRIPTION: [语义描述]
```

响应解析在 `_parse_visual_response()` 中按行解析 `TYPE:`、`TEXT:`、`DESCRIPTION:` 三个字段。

## 完整调用示例

最小可用的独立调用代码：

```python
import base64
from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="your-api-key")

image_data = Path("screenshot.png").read_bytes()
base64_image = base64.b64encode(image_data).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片的内容"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ],
    }
]

response = client.chat.completions.create(
    model="glm-4.6v-flashx",
    messages=messages,
    max_tokens=2000,
)

print(response.choices[0].message.content)
```

## 注意事项

1. **图片格式**：data URI 中固定使用 `image/png`，实际也支持 JPEG 等格式，可按需调整
2. **图片大小**：过大的图片会增加传输和推理耗时，建议关键帧提取时控制分辨率
3. **并发限制**：智谱 API 有请求频率限制，`concurrency` 不宜设置过高
4. **Base64 前缀**：`vision_analysis()` 接收纯 base64 字符串，内部自动拼接 `data:image/png;base64,` 前缀，调用方不需要重复添加
5. **错误处理**：所有 API 错误会转换为项目自定义异常 `VisualAnalysisError`，不会泄露 SDK 内部异常类型
