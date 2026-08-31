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

当前版本已经在 SeeTacloud 云服务器上完整跑通多轮端到端 AgentManager campaign(Stage 1→4),并在两个独立 LLM 后端上验证可复现:DeepSeek(`deepseek-v4-pro/flash/flash-vision-exp`,8-28→8-30)与 OpenAI(`gpt-4.1-mini`,8-31),切换后端无需改任何代码。全程 agent 只根据 validation primary 选择节点,代码不会把 test 标签或 test 指标暴露给 Agent。截至目前最佳 validation primary 为 **0.6053**(相对官方 FM baseline 0.6016,+0.0037;来自 8-29 DeepSeek 运行)。

## 1. 当前状态

已经完成并验证：

- AI Scientist-v2 与 KuaiRand starter kit 的目录和执行接口连接；
- validation-only FM 实验边界；
- DeepSeek 与 OpenAI 两个后端的模型路由(经 `AI_SCIENTIST_*_MODEL` 环境变量切换,无需改代码;上游还保留 Anthropic/Bedrock/Vertex、Ollama 路径,本项目未使用);
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

`primary = (GAUC + nDCG@5) / 2`。上表是 **FM baseline 复现校验**(约等于官方 0.6016),用于确认 pipeline 可复现,不是最终比赛成绩。

在此基础上,agent 自主搜索到的**最佳 validation 节点为 primary 0.6053**(+0.0037 vs FM baseline,来自 8-29 DeepSeek 运行),8-31 换 OpenAI `gpt-4.1-mini` 重跑三次的最佳约 0.6045、未超过它。完整的逐 run 账本见 [`reports/RUN_LOG.md`](reports/RUN_LOG.md)。注意:**最终名次由组委会在 hidden test 上评一次**,参赛期间我们看不到 test 反馈,因此本仓库只自报 validation 指标;官方 FM 在 hidden test 上的参考分为 primary 0.5946。

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
python launch_with_intro.py \
  --config bfts_config_kuairand.yaml \
  --load_ideas ai_scientist/ideas/kuairand_ranking.json \
  --load_code \
  --idea_idx 0 \
  --skip_plots \
  --skip_writeup \
  --skip_review \
  --intro-log-file /tmp/kuairand_agent_baseline.log
```

`launch_with_intro.py` 会在普通输出下方保持 ByteRush 动画，并根据真实启动日志推进进度。
`--intro-log-file` 保存的是不含动画控制字符的纯日志。服务器后台任务或不需要动画时，仍可直接
使用原入口 `launch_scientist_bfts.py`。

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

## 12. 局限性反思与未来工作

### 12.1 已知局限

1. **Agent workflow 仍然偏简单。** 部署配置是最小自主闭环:单 worker、单初始节点、单 seed 复验、搜索树较浅、分支有限,reflect/revise 只做浅层调整,节点间没有跨 run 的记忆或更强的元推理。探索广度不足,是最终结果聚集在 FM baseline 附近(最佳仅 +0.0037)的重要原因之一。

2. **在 KuaiRand-Pure 上的边际收益递减。** 我们把绝大部分时间花在 Pure 的增量调优上。Pure 虽占 primary **100%** 权重,但 FM baseline 已经很强、可提升空间小(oracle 上限 0.8484,FM 已到 0.6016,我们仅 +0.0037),在同一模型家族里继续微调回报很低。更有价值的做法本应是更早地**扩大模型家族搜索空间**,或尝试 bonus 数据集(1k / 27k)换取额外加分。

3. **搜索效率低、token 成本高。** 两轮 campaign 合计约 **17.28M tokens** 才换来 +0.0037 的 primary 提升,单位提升的 token 成本很高。根因是树搜索每个节点都要重新构建上下文,缺乏节点间的缓存/记忆机制。
   > 补充:GPU 峰值占用仅 **~436 MiB / RTX 3080 Ti**,说明在 FM 家族上瓶颈**不是算力**,而是 agent 的模型搜索空间与时间预算。

### 12.2 如果有更多时间

1. **更大显存的卡 + 更多时间 + 扩大候选模型空间,把近年推荐系统架构完整跑到收敛。** 把 agent 的候选从 FM 家族扩展到序列建模与特征交互的代表工作,并给足训练预算跑到收敛:
   - **SASRec**(自注意力序列推荐,2018)、**DIN**(深度兴趣网络,2018)—— 经典基线(DIN 本次已浅层试过,best ≈ 0.6044,但未充分调优、未跑到收敛);
   - **HSTU**(Meta 生成式推荐,2024)、**OneTrans**(WWW 2026,统一特征交互与序列建模的单一 Transformer)、**HyFormer**(2026,长序列建模 + 特征交互统一 backbone)—— 近年 scaling 期的代表架构。
2. 引入**多候选节点 + 多 seed** 的稳健比较,降低单点噪声与偶然性。
3. 为节点间加**缓存 / 记忆机制**,显著降低 token 成本、提高搜索效率。
4. 尝试 bonus 数据集(KuaiRand-1k / 27k)以争取额外加分。
5. 最终模型冻结后再生成 test submission,且全程不向 Agent 返回 test 反馈。

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
