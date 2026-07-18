# FinWise AI — 智能理财助手

一个把财务文档变成**结构化、可核验数据**的网页应用,每次分析后再给出**一句理财建议**,
外加一个**一次性 benchmark**来挑选应用默认使用的模型。

上传**银行对账单 / 发票 / 信用卡账单 / 收据** → 自动识别类型 → 抽取成标准 JSON,
并给出**交易分类、借贷判断、日期与币别标准化、余额对账、重复与异常检测、现金流分析,以及一句理财建议**。

## 快速开始 —— 一条命令

```bash
./run.sh
```

自动建 venv、装依赖,并在 **http://localhost:8000** 打开应用。
- 首次进入是**登录页** —— 选择角色(演示访问,无需注册):
  - **用户** —— 业务视图:上传或试样本 → 抽取、核验、置信度、证据、洞察、导出;历史。隐藏模型/成本/benchmark。
  - **管理员** —— 以上全部 **外加** 模型、成本/延迟、**API Key**,以及 **Model & Benchmark** 页。
- 拖入 PDF/PNG(或选内置样本)→ 点 **Analyze** → 看结果。
- **抽取始终用 vision 模型**(快速/均衡/最佳 = Haiku/Sonnet/Opus)—— 不存在低质量的免费兜底
  (OCR+规则仅是 **benchmark 对照基线**)。
- **API key 三种给法**:启动前 `export ANTHROPIC_API_KEY=sk-ant-...`(运维);应用内右上角 **API Key**
  (**任何用户**都可自带 / BYO,仅本会话、绝不写盘);或管理员为所有人统一配置一次。
- **没配置 key 时**:真实上传显示明确的「添加你的 API key」提示,内置**样本仍用参考数据演示**(有标注)。
- **大文档**:抽取采用流式 + 充足输出预算;若账单太长导致仍被截断,会给出**部分结果 + 明确提示**,而不是崩溃。

> 登录为**演示级**(角色 cookie,非真实多租户账号)—— 见 MVP roadmap。

系统依赖(一次):`brew install tesseract poppler`。

## 应用能做什么

- **用户端零配置**:直接上传 → 分析。**文档类型自动识别**(实在不确定才问;结果里显示
  「已识别为 X · 可改」以纠正误判)—— 无需手动归类,也没有处理模式选择。
- **抽取**成标准 JSON,始终用 **benchmark 推荐模型**(当前 Sonnet —— 实测质量等同 Opus,成本约 40%)。
  管理员仍可选具体模型、文档类型,并查看**成本+延迟**。
- **核验**:字段级**置信度**(高/中/低)+ **来源证据** —— 点击字段即可看它在原文中的出处;
  低置信字段进入待核对;**余额对账**徽章。
- **洞察**:结构化**洞察卡片**(重复扣款、周期扣款、异常支出、大额账单、缺失信息)——
  每张含证据、影响金额、建议操作 —— 外加一句理财建议。
- **分析**:交易分类(彩色 chip)、借贷、日期/币别标准化、现金流拆解。
- **留痕**:**历史**页(SQLite)记录每次分析 —— 筛选、重开、重跑、删除。

## 模型选择 & benchmark(跑一次即可)

benchmark 评估各方法(OCR+规则 baseline vs Haiku/Sonnet/Opus vision)在字段正确率、
交易正确率、对账、延迟、成本、鲁棒性上的表现,并**推荐**应用默认使用的模型。它**由命令行
预跑一次**;**Model & Benchmark 页只只读展示对比**(网页里不编辑数据、不运行 benchmark)。
结果持久化并复用。

```bash
# 预跑一次(vision 方法需要 key),然后打开 /report
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/run_benchmark.sh 20 5          # 每类份数, 预算 $5(4 个模型约 $4–5)
```

它会跑 `M1 + Haiku + Sonnet + Opus`,写入 `benchmark/results/` 并生成报告。想刷新时再跑一次即可。

预算旋钮:`--max-edge`(图像降采样,约 $7.6→$4.8)、`--max-usd`(花费上限)、
`--workers`(并行,约快 4 倍)。需要时才重跑:`--force`。

## 数据准备(一次)

```bash
python -m src.gold.validate 1000 --cache        # 银行交易 gold(1000/1000 对账)
python -m src.generate.credit_card 200          # 合成缺失的信用卡账单
```

## 测试

```bash
python -m pytest tests/ -q
```

## 目录结构

```
run.sh                  一键启动(venv + 依赖 + uvicorn)
webapp/                 FastAPI 产品:main.py + templates/ + static/(自带 CSS/JS)
src/detect.py           文档类型自动识别
src/categorize.py       交易分类(规则 + 可选 LLM)
src/analyze.py          面向 UI 的一站式分析
src/extract/            pdf_text、ocr、rules、llm(vision+text)
src/methods.py          M0-M5 注册表 + recommended_method()
src/postprocess.py      归一化、对账、去重、异常、现金流
src/metrics.py          字段 / 逐行 / 交易 / 对账 评分
src/gold/               从 PDF 文字层解析交易 gold + 对账校验
src/generate/           信用卡账单生成器
benchmark/run.py        并行 benchmark + recommend() + 缓存(recommendation.json)
report/generate.py      结果 -> report.md
```
