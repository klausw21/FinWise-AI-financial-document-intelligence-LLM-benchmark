# Tech Stack · 技术栈

FinWise AI — Intelligent Personal-Finance Assistant & LLM Benchmark.
A Python web product that extracts financial documents into structured JSON, gives a
one-line financial tip per analysis, and benchmarks extraction models.
一个 Python 网页产品:把财务文档抽取成结构化 JSON、每次分析给出一句理财建议,并对抽取模型做 benchmark。

---

## Overview · 总览

| Area · 领域 | Choice · 选型 |
|---|---|
| Language · 语言 | Python 3.11+ |
| Web · 后端/网页 | FastAPI · Uvicorn · Jinja2 (server-rendered) |
| Frontend · 前端 | Vanilla HTML/CSS/JS — self-contained, no Node, no CDN · 纯手写,无 Node/CDN |
| AI / LLM | Anthropic Claude API (`anthropic` SDK) — Opus 4.8 / Sonnet 5 / Haiku 4.5 |
| Docs / OCR | pdfplumber · PyMuPDF · Poppler(pdftotext) · Tesseract(pytesseract) · Pillow |
| Data / Report | pydantic v2 · pandas · matplotlib · rapidfuzz · numpy · reportlab · Faker · tabulate |
| Test / Tooling | pytest · httpx (TestClient) · venv |

---

## Backend & core · 后端与核心

| Technology | Purpose (EN) | 用途 (中文) |
|---|---|---|
| **Python 3.11+** | Language for the whole stack | 全栈语言 |
| **pydantic v2** | Typed schemas for the 5 document types; strict JSON Schema for Claude structured output | 5 类文档的类型化 schema;生成 Claude structured-output 的严格 JSON Schema |
| **pandas** | Benchmark result aggregation, pivots, recommendation | benchmark 结果聚合、透视、推荐计算 |
| **matplotlib** | Benchmark charts (accuracy-vs-cost, robustness) | benchmark 图表(准确率-成本、鲁棒性) |
| **rapidfuzz** | Fuzzy string matching in field-accuracy scoring | 字段正确率评分里的模糊字符串匹配 |
| **numpy** | Image noise generation for the robustness axis | 鲁棒性扰动的图像加噪 |

## Documents · OCR · images · 文档 / OCR / 图像

| Technology | Purpose (EN) | 用途 (中文) |
|---|---|---|
| **pdfplumber / PyMuPDF (fitz)** | PDF text-layer extraction; render PDF → PNG | PDF 文字层抽取;PDF 渲染为 PNG |
| **Poppler (`pdftotext`)** *(system)* | Layout-preserving PDF text → transaction gold parsing | 保留版式的 PDF 取文 → 交易 gold 解析 |
| **Tesseract + pytesseract** *(system + Python)* | OCR on page images (OCR-based methods) | 页面图 OCR(OCR 类方法) |
| **Pillow (PIL)** | Image encode/resize + robustness degradations | 图像编码/缩放 + 鲁棒性降质 |
| **reportlab** | Generate the synthetic credit-card statements (missing type) | 生成缺失的信用卡账单(合成) |

## AI / LLM

| Technology | Purpose (EN) | 用途 (中文) |
|---|---|---|
| **Anthropic Claude API** (`anthropic` SDK) | Vision + text extraction, doc-grounded chat, optional LLM categorization | vision + 文本抽取、文档问答 chat、可选 LLM 分类 |
| **Models** | Opus 4.8 (quality) · Sonnet 5 (main) · Haiku 4.5 (cheap) | Opus 4.8(质量)· Sonnet 5(主力)· Haiku 4.5(便宜) |
| **Structured Outputs** (`output_config.format`) | Force schema-valid JSON output | 强制输出符合 schema 的 JSON |
| **Adaptive thinking** (`thinking: adaptive`, summarized) | Optional reasoning with a shown summary | 可选思考,展示 summarized 摘要 |
| **Cost tracking** | Real `response.usage` → per-call cost | 真实 `response.usage` → 单次成本 |

## Web product · 网页产品

| Technology | Purpose (EN) | 用途 (中文) |
|---|---|---|
| **FastAPI** | HTTP API + page routes (`/`, `/report`, `/api/*`) | HTTP API + 页面路由 |
| **Uvicorn** | ASGI server (launched by `run.sh`) | ASGI 服务器(`run.sh` 启动) |
| **Jinja2** | Server-rendered templates (`base/index/report`) | 服务端渲染模板 |
| **python-multipart** | File-upload handling | 文件上传 |
| **Vanilla CSS** | Light fintech design system + dark mode (CSS variables) | 浅色 fintech 设计系统 + 深色模式(CSS 变量) |
| **Vanilla JS** | Upload/analyze, chat, inline-SVG charts (donut/bars), i18n, theme | 上传/分析、chat、内联 SVG 图表、i18n、主题 |
| **Custom i18n** | EN/ZH toggle (`data-i18n` + dictionary, localStorage) | 中/EN 切换 |

## Testing & tooling · 测试与工具

| Technology | Purpose (EN) | 用途 (中文) |
|---|---|---|
| **pytest** | 29 fast, key-free tests (loaders, gold, metrics, web API, i18n parity) | 29 项免 key 快测(loader/gold/metrics/web/i18n) |
| **httpx** | FastAPI `TestClient` transport | FastAPI `TestClient` |
| **Faker** | Realistic names/merchants for synthetic data | 合成数据的真实姓名/商户 |
| **tabulate** | Markdown tables in the generated report | 报告里的 Markdown 表格 |
| **venv** | Isolated environment (`.venv`, one-command `run.sh`) | 隔离环境(`.venv`,一键 `run.sh`) |

---

## Notable engineering choices · 关键工程取舍

- **No Node / no CDN** — the whole frontend is bundled and works offline.
  前端全部本地打包,离线可用,不依赖 Node/CDN。
- **Transaction gold from the born-digital PDF text layer** — bank-statement labels only
  hold a count, so per-transaction gold is parsed from the PDF and validated by
  balance reconciliation (1000/1000).
  银行对账单 label 只有笔数,逐笔 gold 从 PDF 文字层解析并用余额对账校验(1000/1000)。
- **Benchmark = one-off CLI, read-only report** — run once (`scripts/run_benchmark.sh`),
  results persist with a fingerprint cache; the product uses the recommended model.
  benchmark 命令行一次性预跑,结果带 fingerprint 缓存持久化;`/report` 只读;产品用推荐模型。
- **Parallel benchmark** — `ThreadPoolExecutor` (~4× faster on OCR, more on network-bound LLMs),
  with a soft budget cap.
  benchmark 并行(OCR 约 4×,LLM 更多),带软预算上限。
- **Cost from real usage** — never estimated; image downscale + budget cap as cost levers.
  成本取真实 usage,非估算;图像降采样 + 预算上限作为成本杠杆。

## System prerequisites · 系统依赖

```bash
brew install tesseract poppler      # OCR + PDF text (macOS)
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```
