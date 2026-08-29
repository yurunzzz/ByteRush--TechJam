"""KuaiRand-Pure 任务导向 EDA —— matplotlib 出图版。

复用官方 data.py loader，口径不变。每张图都对应 eda.py 里的一个分析点，
产出 PNG 到 --out-dir，便于放进实验记录 / 报告。

用法:
    python eda_plots.py --data-dir KuaiRand-Pure/data --out-dir eda_figs
依赖: numpy, matplotlib
"""
import argparse, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt

import data as data_module

SPLIT_COLOR = {"train": "#4C72B0", "valid": "#DD8452", "test": "#55A868"}


# ---------- 通用聚合 ----------
def per_user_labels(rows):
    d = collections.defaultdict(list)
    for x in rows:
        d[x[1]].append(x[6])
    return d


def daily(rows):
    day = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        day[x[0]][0] += x[6]
        day[x[0]][1] += 1
    ds = sorted(day)
    return ds, [day[d][1] for d in ds], [day[d][0] / day[d][1] for d in ds]


def field_rate(rows, fi, dur_bucket=False):
    agg = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        key = int(min(9, x[5] // 5000)) if dur_bucket else x[fi]
        agg[key][0] += x[6]
        agg[key][1] += 1
    return agg


# ---------- 各图 ----------
def fig_list_length(splits, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("train", "valid", "test"):
        ll = [len(v) for v in per_user_labels(splits[name]).values()]
        ll = np.clip(ll, 0, np.quantile(ll, 0.99))  # 截 p99 防长尾压扁
        ax.hist(ll, bins=40, alpha=0.5, label=name, color=SPLIT_COLOR[name],
                density=True)
    ax.set(xlabel="每用户曝光数 (list 长度, 截至 p99)", ylabel="密度",
           title="每用户曝光数分布 —— 越长排序越难")
    ax.legend()
    save(fig, out / "01_list_length.png")


def fig_user_pos_rate(splits, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("train", "valid", "test"):
        ur = [sum(v) / len(v) for v in per_user_labels(splits[name]).values()]
        ax.hist(ur, bins=40, alpha=0.5, label=name, color=SPLIT_COLOR[name],
                density=True)
    ax.set(xlabel="每用户 long_view 正例率", ylabel="密度",
           title="每用户正例率分布 —— 0 与 1 两端是 GAUC/nDCG 的死区")
    ax.legend()
    save(fig, out / "02_user_pos_rate.png")


def fig_user_taxonomy(splits, out):
    """混合 / 零正例 / 全正例 三类用户占比 —— 直接决定指标天花板。"""
    cats, mixed, zero, allpos = [], [], [], []
    for name in ("train", "valid", "test"):
        pu = per_user_labels(splits[name])
        z = sum(1 for v in pu.values() if sum(v) == 0)
        a = sum(1 for v in pu.values() if sum(v) == len(v))
        m = len(pu) - z - a
        t = len(pu)
        cats.append(name); mixed.append(m/t); zero.append(z/t); allpos.append(a/t)
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, mixed, label="混合 (进 GAUC)", color="#4C72B0")
    ax.bar(x, zero, bottom=mixed, label="零正例 (nDCG=0)", color="#C44E52")
    ax.bar(x, allpos, bottom=np.array(mixed)+np.array(zero),
           label="全正例 (GAUC 不计)", color="#8C8C8C")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set(ylabel="用户占比", title="用户分类占比 —— 只有混合用户真正贡献 GAUC")
    for i in range(len(cats)):
        ax.text(i, mixed[i]/2, f"{mixed[i]*100:.0f}%", ha="center", color="w")
    ax.legend()
    save(fig, out / "03_user_taxonomy.png")


def fig_daily(splits, out):
    """每日曝光量 + 正例率, 用底色标出 train/valid/test 时段 -> 看漂移。"""
    all_rows = splits["train"] + splits["valid"] + splits["test"]
    ds, vol, rate = daily(all_rows)
    xs = list(range(len(ds)))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(xs, vol, color="#B0B0B0", alpha=0.6)
    ax1.set_ylabel("曝光量", color="#666")
    ax2 = ax1.twinx()
    ax2.plot(xs, rate, "-o", color="#DD8452", lw=2)
    ax2.set_ylabel("long_view 正例率", color="#DD8452")
    # 时段边界
    tr_n = len({d for d in {x[0] for x in splits['train']}})
    va_n = len({x[0] for x in splits['valid']})
    ax1.axvspan(-0.5, tr_n-0.5, color="#4C72B0", alpha=0.06)
    ax1.axvspan(tr_n-0.5, tr_n+va_n-0.5, color="#DD8452", alpha=0.06)
    ax1.axvspan(tr_n+va_n-0.5, len(ds)-0.5, color="#55A868", alpha=0.06)
    ax1.set_xticks(xs); ax1.set_xticklabels([str(d)[4:] for d in ds], rotation=90, fontsize=7)
    ax1.set(xlabel="日期 (蓝=train 橙=valid 绿=test)",
            title="每日曝光量与正例率 —— 蓝→橙的正例率漂移决定要不要 recency 加权")
    save(fig, out / "04_daily_drift.png")


def fig_cold_start(splits, out):
    tr_u = {x[1] for x in splits["train"]}
    tr_v = {x[2] for x in splits["train"]}
    tr_a = {x[3] for x in splits["train"]}
    labels, uu, vv, aa = [], [], [], []
    for name in ("valid", "test"):
        rows = splits[name]; n = len(rows)
        labels.append(name)
        uu.append(sum(1 for x in rows if x[1] not in tr_u) / n)
        vv.append(sum(1 for x in rows if x[2] not in tr_v) / n)
        aa.append(sum(1 for x in rows if x[3] not in tr_a) / n)
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x-w, uu, w, label="未见 user", color="#4C72B0")
    ax.bar(x,   vv, w, label="未见 video", color="#DD8452")
    ax.bar(x+w, aa, w, label="未见 author", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="曝光行占比", title="冷启动 UNK 率 (相对 train) —— 纯 ID FM 的信息盲区")
    for i in range(len(labels)):
        for off, val in ((-w, uu[i]), (0, vv[i]), (w, aa[i])):
            ax.text(i+off, val, f"{val*100:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.legend()
    save(fig, out / "05_cold_start.png")


def fig_feature_rate(splits, out):
    tr = splits["train"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # tab
    agg = field_rate(tr, 4)
    items = sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]
    axes[0].bar([str(k) for k, _ in items], [p/n for _, (p, n) in items], color="#4C72B0")
    axes[0].set(title="tab 上的正例率 (train)", xlabel="tab", ylabel="正例率")
    axes[0].tick_params(axis="x", rotation=45)
    # dur bucket
    agg = field_rate(tr, 5, dur_bucket=True)
    items = sorted(agg.items())
    axes[1].bar([str(k) for k, _ in items], [p/n for _, (p, n) in items], color="#DD8452")
    axes[1].set(title="时长桶 (每5s一桶, 封顶9) 正例率 (train)",
                xlabel="dur bucket", ylabel="正例率")
    fig.suptitle("特征-标签关系 —— 有区分度才值得留 / 做交叉")
    save(fig, out / "06_feature_rate.png")


def fig_popularity_drift(splits, out):
    """author 流行度 train vs valid 散点 —— 偏离对角线=流行度漂移。"""
    def cnt(rows, idx):
        c = collections.Counter(x[idx] for x in rows)
        return c
    ct = cnt(splits["train"], 3)
    cv = cnt(splits["valid"], 3)
    common = [a for a in ct if a in cv]
    if not common:
        return
    xs = np.array([ct[a] for a in common], float)
    ys = np.array([cv[a] for a in common], float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=6, alpha=0.3, color="#4C72B0")
    lim = max(xs.max(), ys.max())
    ax.plot([1, lim], [1, lim], "--", color="#C44E52", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="train 曝光次数", ylabel="valid 曝光次数",
           title="author 流行度漂移 (log-log) —— 偏离红线=分布变了")
    save(fig, out / "07_author_popularity_drift.png")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("KuaiRand-Pure/data"))
    ap.add_argument("--out-dir", type=Path, default=Path("eda_figs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 中文字体: 按文件路径注册第一个存在的 CJK 字体, 缺失则回退(标题变方块但不报错)
    from matplotlib import font_manager
    for path in ("/Library/Fonts/Arial Unicode.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                matplotlib.rcParams["font.sans-serif"] = [
                    font_manager.FontProperties(fname=path).get_name()]
                break
            except Exception:
                continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("加载官方切分 ...")
    splits = data_module.load(str(args.data_dir))
    print("出图中 ...")
    fig_list_length(splits, args.out_dir)
    fig_user_pos_rate(splits, args.out_dir)
    fig_user_taxonomy(splits, args.out_dir)
    fig_daily(splits, args.out_dir)
    fig_cold_start(splits, args.out_dir)
    fig_feature_rate(splits, args.out_dir)
    fig_popularity_drift(splits, args.out_dir)
    print(f"完成 -> {args.out_dir}/  (共 7 张图)")


if __name__ == "__main__":
    main()
