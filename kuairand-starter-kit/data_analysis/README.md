# data_analysis —— KuaiRand-Pure 任务导向 EDA

针对 `long_view` 用户内排序任务（主指标 ½(GAUC + nDCG@5)）的探索性数据分析。
所有脚本复用上级目录的官方 `data.py` loader，口径不变，不触碰 `evaluate.py` / 切分 / 标签。

## 文件

| 文件 | 作用 |
|------|------|
| `eda.py` | 文本报告：规模、用户级标签结构、每用户 list 长度、冷启动 UNK 率、时间漂移、tab/时长的正例率 |
| `eda_plots.py` | 7 张 matplotlib 图（对应上表各分析点），输出到 `eda_figs/` |
| `eda_report.html` | 一页诊断报告：指标上下界阶梯、用户构成、建模优先级 |
| `eda_figs/` | `eda_plots.py` 生成的 PNG |

## 用法

`--data-dir` 默认指向 `../KuaiRand-Pure/data`（官方数据位置）；数据在别处时显式传入。

```bash
cd kuairand-starter-kit/data_analysis
python eda.py                         # 文本报告
python eda_plots.py                   # 重新生成 eda_figs/ 下 7 张图
python eda.py --raw-log-scan          # 额外扫原始日志, 列出全部可用反馈列(多任务原料)
```

## 主要结论（test split，来自真数据，已与官方 `baseline_scores.json` 交叉验证）

- **nDCG@5 天花板 ≈ 0.729**（非 1.0）：27.1% 用户全负，nDCG 恒为 0 且计入平均；仅 63.7% 用户进 GAUC。
- **冷启动几乎不存在**：未见 video/author ≈ 0.01% → 加内容/兜底特征 ROI 极低。
- **tab 是强信号**（正例率 3.7%–45.6%）；时间漂移温和。
- **优先级**：pointwise logloss 与排序指标不一致 → 先上 pairwise/BPR，主攻 GAUC 一侧空间。
