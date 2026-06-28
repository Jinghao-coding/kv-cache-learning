# AGENTS.md — KV Cache Learning Project

## 项目概述

通过 **理论 + 从零代码实现** 的方式，系统理解 LLM 推理中的 KV Cache 技术。

## 目录结构

```
kv-cache-learning/
├── AGENTS.md              # 本文件：项目规范与进度
├── README.md              # 项目入口与学习路线
├── pyproject.toml         # Python 项目配置
├── .gitignore             # Git 忽略规则
├── .venv/                 # 虚拟环境（不入库）
├── docs/                  # 理论笔记（Markdown）
│   └── module{N}_{name}.md
├── src/kvcache/           # 代码实现
│   ├── __init__.py
│   ├── modules/           # 每个学习模块的可运行代码
│   │   ├── module0_attention.py
│   │   ├── module1_kv_cache_basic.py
│   │   └── module2_memory_calc.py
│   └── utils/             # 共享工具（计算、可视化等）
├── tests/                 # 单元测试
└── examples/              # 可直接运行的示例脚本
```

## 学习模块进度

| # | 模块 | 文档 | 代码 | 状态 |
|---|------|------|------|------|
| 0 | 注意力机制回顾 | - | [module0](src/kvcache/modules/module0_attention.py) | ✅ 完成 |
| 1 | KV Cache 基础原理 | - | [module1](src/kvcache/modules/module1_kv_cache_basic.py) | ✅ 完成 |
| 2 | KV Cache 内存计算 | - | [module2](src/kvcache/modules/module2_memory_calc.py) | ✅ 完成 |
| 3 | MQA / GQA | - | - | ⬜ 待开始 |
| 4 | PagedAttention (vLLM) | - | - | ⬜ 待开始 |
| 5 | KV Cache 量化 | - | - | ⬜ 待开始 |
| 6 | KV Cache 驱逐/压缩 | - | - | ⬜ 待开始 |
| 7 | 框架实现对比 | - | - | ⬜ 待开始 |

## 技术栈

- **语言**: Python 3.10+
- **核心依赖**: NumPy（核心计算）、PyTorch（对照验证）、Rich（终端可视化）
- **测试**: pytest
- **包管理**: uv

## 运行方式

```bash
source .venv/bin/activate
python -m kvcache.modules.module0_attention
python -m kvcache.modules.module1_kv_cache_basic
python -m kvcache.modules.module2_memory_calc
```

## 编码规范

- 每个模块代码**自包含**，可独立运行 `python -m kvcache.modules.module{N}_{name}`
- 理论文档紧跟代码实现，文档中引用代码行号
- 核心公式必须在模块 docstring 中给出数学表达式
- 不添加不必要的注释，代码自解释
- 类型注解必须完备
- demo() 函数必须使用 Rich 输出格式化的表格/面板

## Git 规范

- commit message 格式: `type(scope): message`
- type: `feat`(新模块), `docs`(文档), `fix`(修复), `chore`(构建/配置), `refactor`(重构)
- 每个模块完成后自动 commit + push
- 不提交 .venv/、__pycache__/、*.pyc 等
