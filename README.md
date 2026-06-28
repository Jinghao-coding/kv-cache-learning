# KV Cache Learning

通过 **理论 + 从零代码实现** 的方式，系统理解 LLM 推理中的 KV Cache 技术。

## 快速开始

```bash
# 创建虚拟环境并安装依赖
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行任意学习模块
python -m kvcache.modules.module0_attention       # 注意力机制回顾
python -m kvcache.modules.module1_kv_cache_basic  # KV Cache 基础原理
python -m kvcache.modules.module2_memory_calc     # KV Cache 内存计算
```

## 学习路线

| # | 模块 | 核心内容 | 代码 | 状态 |
|---|------|---------|------|------|
| 0 | 注意力机制回顾 | Scaled Dot-Product Attention、Multi-Head Attention（纯 NumPy 实现） | [module0_attention.py](src/kvcache/modules/module0_attention.py) | ✅ |
| 1 | KV Cache 基础原理 | Prefill/Decode 两阶段、朴素生成 vs KV Cache 生成对比 | [module1_kv_cache_basic.py](src/kvcache/modules/module1_kv_cache_basic.py) | ✅ |
| 2 | KV Cache 内存计算 | MHA/MQA/GQA 内存公式、主流模型 KV Cache 大小估算 | [module2_memory_calc.py](src/kvcache/modules/module2_memory_calc.py) | ✅ |
| 3 | MQA / GQA | Multi-Query Attention、Grouped-Query Attention 的 KV Cache 缩减 | - | 🔜 |
| 4 | PagedAttention | vLLM 分页 KV Cache 原理与 Block 管理 | - | 🔜 |
| 5 | KV Cache 量化 | INT8/INT4 KV Cache 量化方案 | - | 🔜 |
| 6 | KV Cache 驱逐/压缩 | StreamingLLM、H2O、Attention Sinks | - | 🔜 |
| 7 | 框架实现对比 | vLLM / TensorRT-LLM / SGLang KV Cache 管理对比 | - | 🔜 |

## 项目结构

```
kv-cache-learning/
├── AGENTS.md              # 项目规范与进度追踪
├── pyproject.toml         # Python 项目配置
├── src/kvcache/
│   └── modules/           # 每个学习模块可独立运行
├── docs/                  # 理论笔记
├── tests/                 # 单元测试
└── examples/              # 可直接运行的示例
```
