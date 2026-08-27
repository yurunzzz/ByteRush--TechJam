# ByteRush：面向 KuaiRand 推荐系统的自动机器学习研究 Agent

本仓库是 TikTok TechJam 题目 **Autonomous Machine Learning Research Agent for Recommender Systems** 的团队实现。项目将 [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) 的 AgentManager 与组委会提供的 KuaiRand-Pure Factorization Machine（FM）baseline 连接起来，使大语言模型能够自动完成：

```text
提出实验方案并生成代码
→ 在隔离工作区执行代码
→ 调用受控的 FM validation 实验接口
→ 解析 GAUC、nDCG@5 和 primary
→ 将结果写入搜索树节点
→ 复现实验并比较节点
→ 自动选择最佳实现并保存 checkpoint
```

当前版本已经在 SeeTacloud 云服务器上完整跑通 Stage 1 AgentManager baseline。代码不会把 test 标签或 test 指标暴露给 Agent。

## 1. 当前状态

已经完成并验证：

- AI Scientist-v2 与 KuaiRand starter kit 的目录和执行接口连接；
- validation-only FM 实验边界；
- DeepSeek V4、OpenAI 和 Ollama 的模型路由；
- DeepSeek 结构化工具调用与指标解析；
- LLM 自动生成代码、V2 自动执行、自动分析结果；
- journal 节点创建、单 seed 复验和最佳节点选择；
- 搜索树 HTML、实验数据和 checkpoint 保存；
- `data.py`、`evaluate.py` 运行前后 SHA-256 完整性检查；
- smoke test 和完整 AgentManager Stage 1 流程。

最后一次成功验证的 validation 指标为：

| 指标 | 数值 |
|---|---:|
| GAUC | 0.667133391 |
| nDCG@5 | 0.535805702 |
| primary | 0.601469547 |
| best epoch | 7 |

`primary = (GAUC + nDCG@5) / 2`。这些数字用于确认 pipeline 可复现，不是最终比赛成绩。

## 2. 比赛任务口径

| 项目 | 约定 |
|---|---|
| 任务 | 用户内排序：对每位用户已经曝光的视频重新排序 |
| 相关性标签 | `long_view`（0/1） |
| 指标 | GAUC、nDCG@5 |
| 主指标 | `(GAUC + nDCG@5) / 2`，越高越好 |
| 数据划分 | 官方 chronological train / valid / test |
| Agent 可见反馈 | 仅 validation |
| 禁止事项 | test 反馈、修改官方 evaluator、把 validation/test 标签作为特征、改变行顺序 |

`evaluate.py` 是唯一评分口径。当前研究循环必须根据 validation primary 选择节点；test 只能在最终模型冻结后用于生成提交文件，不能返回给 Agent。

## 3. 仓库结构

```text
ByteRush/
├── ai_scientist/                         # AI Scientist-v2 核心代码
│   ├── ideas/
│   │   ├── kuairand_ranking.json         # Stage 1 预生成研究任务
│   │   └── kuairand_ranking.py           # V2 可执行 FM baseline 起始代码
│   ├── llm.py                            # LLM 客户端与模型路由
│   └── treesearch/                       # AgentManager、Interpreter、搜索树
├── kuairand-starter-kit/
│   ├── data.py                           # 官方数据读取和划分；受保护
│   ├── evaluate.py                       # 官方指标实现；受保护
│   ├── baseline.py                       # random / popularity / FM baseline
│   ├── run_fm_experiment.py              # validation-only 可信实验接口
│   ├── fm_experiment_config.json         # FM 实验配置
│   ├── submit.py                         # 提交文件生成与检查
│   ├── AGENT_FM_INTERFACE.md             # Agent 与 FM 接口契约（运行必需）
│   └── kuairand_ranking.md               # Agent 研究题目输入（运行必需）
├── bfts_config_kuairand.yaml              # KuaiRand Stage 1 Agent 配置
├── run_v2_fm_smoke.py                    # 不调用 LLM 的端到端 smoke test
├── requirements_kuairand.txt              # 精简且已验证的依赖
└── launch_scientist_bfts.py               # AgentManager 启动入口
```

以下内容不会上传 GitHub：KuaiRand 原始数据和压缩包、API Key、`.env`、`experiments/`、`deployment_runs/`、缓存和模型 checkpoint。

## 4. 环境与数据准备

推荐 Linux、Python 3.11/3.12 和 CUDA PyTorch。FM 本身也可以在 CPU 上运行。

```bash
cd /path/to/ByteRush
python -m pip install -r requirements_kuairand.txt
python -m pip check
```

云服务器的 CUDA PyTorch 通常由镜像提供，不建议在未确认 CUDA 版本时覆盖安装。

数据不进入 Git，应放在：

```text
kuairand-starter-kit/KuaiRand-Pure/data/
```

如需下载：

```bash
cd kuairand-starter-kit
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
cd ..
test -d kuairand-starter-kit/KuaiRand-Pure/data && echo "data ready"
```

## 5. 配置 LLM API

仓库不保存 API Key。凭据必须在启动 Agent 的同一个 shell 或 tmux 会话中加载。

### DeepSeek（当前默认）

```bash
export DEEPSEEK_API_KEY="your-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_THINKING="disabled"
```

保持 `DEEPSEEK_THINKING=disabled`：V2 使用强制工具调用解析结构化指标，DeepSeek V4 thinking 模式与该工具选择不兼容。

| Agent 角色 | 默认模型 |
|---|---|
| 代码生成/研究 | `deepseek-v4-pro` |
| 反馈/报告/总结/节点选择 | `deepseek-v4-flash` |
| 视觉反馈 | `deepseek-v4-flash-vision-exp` |

安全检查（不会打印 Key）：

```bash
python -c "import os; print('DeepSeek key loaded:', bool(os.getenv('DEEPSEEK_API_KEY')))"
```

### 切换到 OpenAI

不需要修改 Python 或 YAML：

```bash
export OPENAI_API_KEY="your-key"
export AI_SCIENTIST_CODE_MODEL="gpt-4.1-mini"
export AI_SCIENTIST_FEEDBACK_MODEL="gpt-4.1-mini"
export AI_SCIENTIST_REPORT_MODEL="gpt-4.1-mini"
export AI_SCIENTIST_VLM_MODEL="gpt-4.1-mini"
export AI_SCIENTIST_SUMMARY_MODEL="gpt-4.1-mini"
export AI_SCIENTIST_SELECT_MODEL="gpt-4.1-mini"
```

兼容 OpenAI API 的网关还可设置 `OPENAI_BASE_URL`。取消六个 `AI_SCIENTIST_*_MODEL` 环境变量即可恢复 DeepSeek 默认值。

## 6. 推荐使用 tmux

```bash
tmux new-session -A -s byterush
```

在 tmux 中加载 API Key 后再运行 Agent。安全离开会话时按 `Ctrl+B`，松开后按 `D`。不要在唯一窗口中输入 `exit` 或按 `Ctrl+D`，否则会话及临时环境变量会消失。tmux 消失不会删除磁盘文件，但需要重新加载 Key。

## 7. 运行 smoke test

smoke test 不调用 LLM，用于验证 V2 Interpreter、FM validation-only 接口、指标结构、primary 计算、受保护文件哈希和 `experiment_data.npy`。

```bash
cd /path/to/ByteRush
python run_v2_fm_smoke.py
```

成功时最后出现：

```text
V2_FM_SMOKE_RESULT {"status": "success", ...}
```

默认产物位于 `deployment_runs/v2_fm_baseline/`。

## 8. 运行完整 AgentManager Baseline

先确认至少一个 Key 已加载：

```bash
python -c "import os; print(bool(os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')))"
```

然后从仓库根目录运行：

```bash
python launch_scientist_bfts.py \
  --config bfts_config_kuairand.yaml \
  --load_ideas ai_scientist/ideas/kuairand_ranking.json \
  --load_code \
  --idea_idx 0 \
  --skip_plots \
  --skip_writeup \
  --skip_review \
  2>&1 | tee /tmp/kuairand_agent_baseline.log
```

不要传入 `--add_dataset_ref`。当前配置是最小自主闭环：单 worker、单初始节点、一个 seed 复验，不生成论文或 review。它验证 Agent pipeline 的可靠性，不负责直接提升比赛指标；后续应保留它作为回归测试。

## 9. 成功判据和产物

日志应依次出现：

```text
MinimalAgent: Getting plan and code
MinimalAgent: Draft complete
Running code
Parsing execution results
Added result node to journal
Stage ... completed: found working implementation
Starting multi-seed eval...
Selected node ...
Stage ... multi-seed eval done.
Saving checkpoint to ...
```

单 seed 时出现以下内容是正常的：

```text
Skipping seed plot aggregation: at least two successful seed runs are required.
```

运行目录：

```text
experiments/<timestamp>_kuairand_fm_validation_baseline_attempt_0/
```

| 产物 | 用途 |
|---|---|
| `0-kuairand/process_*/runfile.py` | LLM 生成并执行的代码 |
| `logs/0-kuairand/experiment_results/*/experiment_data.npy` | Agent 可见的 validation 结果 |
| `logs/0-kuairand/manager.pkl` | AgentManager 状态 |
| `logs/0-kuairand/unified_tree_viz.html` | 搜索树可视化 |
| `logs/0-kuairand/stage_*/checkpoint.pkl` | 阶段 checkpoint |
| `token_tracker*.json` | LLM 调用与 token 记录 |

## 10. Validation-only FM 接口

独立运行一个受控实验：

```bash
cd kuairand-starter-kit
python run_fm_experiment.py \
  --config fm_experiment_config.json \
  --data-dir KuaiRand-Pure/data \
  --output-dir experiments/fm_validation_baseline
```

stdout 的机器可读记录以 `AI_SCIENTIST_RESULT ` 开头。成功结果只包含 validation 指标、checkpoint、运行时间、seed、行数和受保护文件哈希。

允许配置的实验参数只有 `seed`、`embedding_dim`、`learning_rate`、`l2`、`batch_size`、`max_epochs`、`early_stopping_patience`、`min_delta` 和安全的 `experiment_name`。未知字段以及替换 label、split、evaluator 或 data loader 的尝试会被拒绝。详细契约见 `kuairand-starter-kit/AGENT_FM_INTERFACE.md`。

## 11. 常见问题

### tmux 显示 `no sessions`

```bash
tmux new-session -A -s byterush
```

然后重新加载 API Key。代码和实验文件仍保存在磁盘上。

### `OPENAI_API_KEY environment variable is not set`

说明某个 Agent 角色仍配置为 GPT，或当前 shell 没加载 Key。检查实际模型：

```bash
python - <<'PY'
from omegaconf import OmegaConf
c = OmegaConf.load("bfts_config_kuairand.yaml")
print("code:", c.agent.code.model)
print("feedback:", c.agent.feedback.model)
print("summary:", c.agent.summary.model)
print("select:", c.agent.select_node.model)
PY
```

### 找不到 `input` 或 `experiment_data.npy`

必须从仓库根目录使用本文命令启动，不要直接在 `experiments/.../process_*` 中运行生成代码。

### 查看实时日志

```bash
tail -f /tmp/kuairand_agent_baseline.log
```

## 12. 后续研究方向

1. Stage 2：自动搜索 FM 超参数；
2. Stage 3：pairwise/listwise 损失、历史序列、多任务、观看时长建模；
3. 增加多个候选节点和多 seed 稳健比较；
4. 为 Stage 3 建立受控 model-plugin 接口；
5. 最终模型冻结后再生成 test submission，且不向 Agent 返回 test 反馈。

## 13. 上传 GitHub 前检查

```bash
git status --short --untracked-files=all
git diff --check
python -m py_compile \
  ai_scientist/llm.py \
  ai_scientist/treesearch/agent_manager.py \
  ai_scientist/treesearch/backend/backend_openai.py \
  ai_scientist/treesearch/interpreter.py \
  ai_scientist/treesearch/parallel_agent.py \
  ai_scientist/ideas/kuairand_ranking.py \
  kuairand-starter-kit/run_fm_experiment.py
```

确认 Git 状态中没有 API Key、`.env`、KuaiRand 数据、实验目录、checkpoint、日志、提交 CSV、SSH 密码或个人绝对路径配置。

## 14. 上游项目与许可证

本项目基于 SakanaAI 的 AI Scientist-v2，并保留其许可证与负责任使用要求。系统会执行 LLM 生成的代码，只应在隔离且受控的环境中运行。任何论文或报告应按上游许可证要求披露 AI Scientist 的使用。

- 上游仓库：<https://github.com/SakanaAI/AI-Scientist-v2>
- 论文：<https://arxiv.org/abs/2504.08066>
- KuaiRand：<https://kuairand.com/>
