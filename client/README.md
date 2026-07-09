# 投标串标检测客户端

调用服务端 API 进行 PDF 文档串标围标检测。

## 环境要求

- Python 3.8+
- `requests` 库

```bash
pip install requests
```

## 配置

编辑 `config.json`，修改 `server` 地址为服务端的 IP 和端口：

```json
{
    "server": "http://192.168.1.100:8000",
    "content_similarity": true,
    "use_gpu": false,
    "poll_interval": 5
}
```

## 使用方法

### 1. 准备文件

将需要检测的 PDF 文件放入 `input/` 文件夹。

### 2. 运行检测

```bash
python run.py
```

### 3. 查看结果

检测完成后，结果自动保存到 `output/` 文件夹：

- `result_20260709_143025.json` — 完整检测结果
- `result_20260709_143025.pdf` — 检测报告（如服务端支持）

每次运行生成带时间戳的文件名，**永不覆盖**。

## 文件夹结构

```
client/
├── run.py          # 客户端脚本
├── config.json     # 配置文件（修改 server 地址即可）
├── README.md       # 本文件
├── input/          # 放入要检测的 PDF
└── output/         # 检测结果自动保存到这里
```
