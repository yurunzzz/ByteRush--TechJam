# data_analysis —— KuaiRand-Pure 任务导向 EDA

针对 `long_view` 用户内排序任务（主指标 ½(GAUC + nDCG@5)）的探索性数据分析。
所有脚本复用上级目录的官方 `data.py` loader，口径不变，不触碰 `evaluate.py` / 切分 / 标签。

## 文件

| 文件 | 作用 |
|------|------|
| `eda.py` | 文本报告：规模、用户级标签结构、每用户 list 长度、冷启动 UNK 率、时间漂移、tab/时长的正例率 |
| `eda_plots.py` | 7 张 matplotlib 图（对应上表各分析点），输出到 `eda_figs/` |
| `eda_report.html` | 一页诊断报告（给**人**看）：指标上下界阶梯、用户构成、建模优先级 |
| `eda_figs/` | `eda_plots.py` 生成的 PNG |
| `eda_summary.json` | 机器可读事实（给 **agent** 看）：population 级统计、天花板、冷启动、tab 信号 + `priors_for_agent`。只含描述性量，无 per-id 表（防泄漏）|

## 用法

`--data-dir` 默认指向 `../KuaiRand-Pure/data`（官方数据位置）；数据在别处时显式传入。

```bash
cd kuairand-starter-kit/data_analysis
python eda.py                         # 文本报告
python eda_plots.py                   # 重新生成 eda_figs/ 下 7 张图
python eda.py --raw-log-scan          # 额外扫原始日志, 列出全部可用反馈列(多任务原料)
python eda.py --emit-json             # 生成 eda_summary.json (给 agent 用的机器可读事实)
```

## 给 agent 用（AI-Scientist-v2 集成）

agent 读的是 `task_desc`（来自 `ai_scientist/ideas/kuairand_ranking.json` → `idea.md` → 提示词），
不读 HTML/PNG。EDA 结论通过两条通道进入 agent：

1. **`eda_summary.json`** —— 机器可读事实；agent 的 input 根即本 starter-kit，故路径为
   `input/data_analysis/eda_summary.json`，种子代码或工具可 `json.load`。
2. **idea JSON 的 `Abstract` + `Experiments`** —— 注入了一段防泄漏的「数据先验」：冷启动可忽略→别加内容特征；
   nDCG 结构性封顶→主攻 GAUC；tab 强信号→做 tab×user/video 交叉；pointwise 与排序指标不一致→先上 pairwise/BPR；
   任何时序/统计特征必须因果构造。`Abstract` 全程可见，`Experiments` 在 Stage 3 渲染为 Experiment Plan。

先验只给**方向/假设**，不提供可当特征的 per-id 统计表 —— 否则会被 agent 直接当特征贴入，违反因果/防泄漏约束。

## 主要结论（test split，来自真数据，已与官方 `baseline_scores.json` 交叉验证）

- **nDCG@5 天花板 ≈ 0.729**（非 1.0）：27.1% 用户全负，nDCG 恒为 0 且计入平均；仅 63.7% 用户进 GAUC。
- **冷启动几乎不存在**：未见 video/author ≈ 0.01% → 加内容/兜底特征 ROI 极低。
- **tab 是强信号**（正例率 3.7%–45.6%）；时间漂移温和。
- **优先级**：pointwise logloss 与排序指标不一致 → 先上 pairwise/BPR，主攻 GAUC 一侧空间。
