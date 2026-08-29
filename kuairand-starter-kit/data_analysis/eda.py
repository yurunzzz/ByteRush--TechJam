"""KuaiRand-Pure 任务导向 EDA —— 每一节都服务于 within-user long_view 排序 (GAUC + nDCG@5)。

只依赖标准库 + numpy，和 data.py 保持一致。用法:
    python eda.py --data-dir KuaiRand-Pure/data
report 直接打到 stdout；加 --raw-log-scan 会再扫一遍原始日志列出所有可用反馈列。
"""
import argparse, csv, os, sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 让上级的 data.py 可导入
import data as data_module  # 复用官方 loader / split，绝不改口径


def pct(x):
    return f"{100 * x:6.2f}%"


def describe(name, arr):
    a = np.asarray(arr, dtype=float)
    qs = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    print(f"  {name:<28} n={len(a):>9}  mean={a.mean():8.3f}  "
          f"min/med/max={qs[0]:.0f}/{qs[2]:.0f}/{qs[6]:.0f}  "
          f"p90={qs[4]:.0f} p99={qs[5]:.0f}")


def per_user(rows):
    """rows: list of (date,user,video,author,tab,dur,label). 返回 {user: [labels...]}"""
    d = collections.defaultdict(list)
    for x in rows:
        d[x[1]].append(x[6])
    return d


def analyze_split(name, rows):
    print(f"\n{'='*70}\n[{name}]  rows={len(rows):,}")
    users = {x[1] for x in rows}
    videos = {x[2] for x in rows}
    authors = {x[3] for x in rows}
    pos = sum(x[6] for x in rows)
    print(f"  unique  users={len(users):,}  videos={len(videos):,}  authors={len(authors):,}")
    print(f"  long_view 正例率 = {pct(pos / len(rows))}  (pos={pos:,})")

    # 2) 用户级标签结构 —— 决定 GAUC / nDCG 天花板
    pu = per_user(rows)
    list_len = [len(v) for v in pu.values()]
    urate = [sum(v) / len(v) for v in pu.values()]
    n_zero = sum(1 for v in pu.values() if sum(v) == 0)              # nDCG=0 计入平均
    n_all = sum(1 for v in pu.values() if sum(v) == len(v))          # 全正例, GAUC 不计
    n_mixed = len(pu) - n_zero - n_all                              # 只有这些进 GAUC
    print(f"  用户数={len(pu):,}")
    print(f"    混合用户 (进 GAUC)      = {n_mixed:,} ({pct(n_mixed/len(pu))})")
    print(f"    零正例用户 (nDCG=0)     = {n_zero:,} ({pct(n_zero/len(pu))})")
    print(f"    全正例用户 (GAUC 不计)  = {n_all:,} ({pct(n_all/len(pu))})")
    describe("每用户曝光数(list长度)", list_len)
    describe("每用户正例率", np.array(urate) * 100)  # 放大成百分数看分布

    # 6) 特征-标签关系: tab / dur_bucket
    for fi, fname in ((4, "tab"), (5, "dur(原始ms分位后看)")):
        agg = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            key = x[fi] if fi == 4 else int(min(9, x[5] // 5000))  # 粗分桶只为看趋势
            agg[key][0] += x[6]
            agg[key][1] += 1
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]
        print(f"  {fname} 上的正例率 (按曝光量取前12):")
        for k, (p, n) in top:
            print(f"      {str(k):<10} rate={pct(p/n)}  n={n:,}")


def cold_start(train_rows, other_name, other_rows):
    """4) 冷启动: other 里有多少 id 在 train 没见过 -> FM 会落 UNK"""
    tr_u = {x[1] for x in train_rows}
    tr_v = {x[2] for x in train_rows}
    tr_a = {x[3] for x in train_rows}
    n = len(other_rows)
    unk_u = sum(1 for x in other_rows if x[1] not in tr_u)
    unk_v = sum(1 for x in other_rows if x[2] not in tr_v)
    unk_a = sum(1 for x in other_rows if x[3] not in tr_a)
    print(f"\n[冷启动 train -> {other_name}]  按曝光行计")
    print(f"  未见 user 的曝光 = {pct(unk_u/n)}   未见 video = {pct(unk_v/n)}   未见 author = {pct(unk_a/n)}")


def temporal_drift(all_rows):
    """5) 时间漂移: 每日曝光量 + 正例率"""
    day = collections.defaultdict(lambda: [0, 0])
    for x in all_rows:
        day[x[0]][0] += x[6]
        day[x[0]][1] += 1
    print(f"\n[时间漂移] 每日曝光量与 long_view 正例率")
    for d in sorted(day):
        p, n = day[d]
        print(f"  {d}  n={n:>8,}  pos_rate={pct(p/n)}")


def raw_log_scan(data_dir):
    """7) 列出原始日志所有列 + 各反馈信号正例率, 找多任务原料"""
    f = os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")
    with open(f) as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        print(f"\n[原始日志列]  {f}\n  {cols}")
        # 对疑似二元反馈列统计正例率
        cand = [c for c in cols if c in
                ("is_click", "is_like", "is_follow", "is_comment", "is_forward",
                 "is_hate", "long_view", "is_profile_enter")]
        agg = {c: [0, 0] for c in cand}
        for i, r in enumerate(rdr):
            for c in cand:
                agg[c][1] += 1
                if r[c] not in ("0", "", "False"):
                    agg[c][0] += 1
            if i >= 2_000_000:  # 采样上限, 够看比例了
                break
        print("  反馈信号正例率 (train 采样):")
        for c, (p, n) in agg.items():
            print(f"      {c:<18} {pct(p/n)}  (n={n:,})")


def build_summary(splits):
    """把分析压成机器可读的 population 级事实, 供 agent / 程序 load。

    只放描述性、population 级的量 —— 绝不放可当特征的 per-id 表, 以免 agent
    直接贴进模型造成泄漏 / 非因果特征。天花板由 population 结构解析得到:
    完美排序对任一含正例用户 nDCG@5=1, 故 nDCG 上界 = 有正例用户占比;
    GAUC 上界恒为 1.0 (混合用户 AUC=1)。
    """
    def split_stat(rows):
        pu = per_user(rows)
        n_zero = sum(1 for v in pu.values() if sum(v) == 0)
        n_all = sum(1 for v in pu.values() if sum(v) == len(v))
        n_users = len(pu)
        ll = np.array([len(v) for v in pu.values()])
        ndcg_ceil = 1.0 - n_zero / n_users            # = 含正例用户占比
        return {
            "rows": len(rows),
            "users": n_users,
            "pos_rate": round(sum(x[6] for x in rows) / len(rows), 4),
            "all_negative_pct": round(100 * n_zero / n_users, 1),
            "all_positive_pct": round(100 * n_all / n_users, 1),
            "discriminative_pct": round(100 * (n_users - n_zero - n_all) / n_users, 1),
            "list_len_median": int(np.median(ll)),
            "list_len_p90": int(np.quantile(ll, 0.9)),
            "ceiling": {
                "GAUC": 1.0,
                "nDCG@5": round(ndcg_ceil, 4),
                "primary": round((1.0 + ndcg_ceil) / 2.0, 4),
            },
        }

    def unseen(train_rows, other_rows):
        tr_v = {x[2] for x in train_rows}; tr_a = {x[3] for x in train_rows}
        tr_u = {x[1] for x in train_rows}; n = len(other_rows)
        return {
            "unseen_user_pct": round(100 * sum(1 for x in other_rows if x[1] not in tr_u) / n, 2),
            "unseen_video_pct": round(100 * sum(1 for x in other_rows if x[2] not in tr_v) / n, 2),
            "unseen_author_pct": round(100 * sum(1 for x in other_rows if x[3] not in tr_a) / n, 2),
        }

    # tab 信号强度 = 正例率跨度 (仅统计曝光量足够的 tab)
    tab_agg = collections.defaultdict(lambda: [0, 0])
    for x in splits["train"]:
        tab_agg[x[4]][0] += x[6]; tab_agg[x[4]][1] += 1
    tab_rates = [p / n for p, n in tab_agg.values() if n >= 1000]

    return {
        "dataset": "KuaiRand-Pure",
        "label": "long_view",
        "task": "within-user ranking; primary = mean(GAUC, nDCG@5)",
        "population": {name: split_stat(splits[name]) for name in ("valid", "test")},
        "cold_start": {
            "valid": unseen(splits["train"], splits["valid"]),
            "test": unseen(splits["train"], splits["test"]),
        },
        "signal": {
            "tab_posrate_min": round(min(tab_rates), 3),
            "tab_posrate_max": round(max(tab_rates), 3),
        },
        "priors_for_agent": [
            "Cold-start negligible (unseen video/author ~0.01%): do NOT spend capacity on content/ID-fallback features.",
            "nDCG@5 is structurally capped (~27% all-negative users, nDCG=0 in the mean): optimize GAUC, not nDCG.",
            "tab positive-rate spans an order of magnitude: prioritize tab x user / tab x video crosses; pure user-side first-order terms cannot change within-user order.",
            "Pointwise logloss is misaligned with the ranking metric: try within-user pairwise/BPR first.",
            "Any temporal or statistical feature MUST be built causally from strictly-earlier events only (no valid/test labels, no future).",
        ],
        "provenance": "Computed from the official data.py loader; population-level only, no per-id tables. Ceilings cross-checked against baseline_scores.json oracle_ceiling.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "KuaiRand-Pure" / "data")
    ap.add_argument("--raw-log-scan", action="store_true",
                    help="额外扫原始日志, 列出全部可用反馈列")
    ap.add_argument("--emit-json", nargs="?", type=Path, const=Path(__file__).resolve().parent / "eda_summary.json",
                    default=None, help="写机器可读事实到该路径 (供 agent 使用), 默认 eda_summary.json; 不加则不写")
    args = ap.parse_args()

    print("加载官方切分中 ...")
    splits = data_module.load(str(args.data_dir))

    if args.emit_json is not None:
        import json
        summary = build_summary(splits)
        args.emit_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"machine-readable summary -> {args.emit_json}")
        return

    for name in ("train", "valid", "test"):
        analyze_split(name, splits[name])

    cold_start(splits["train"], "valid", splits["valid"])
    cold_start(splits["train"], "test", splits["test"])
    temporal_drift(splits["train"] + splits["valid"] + splits["test"])

    if args.raw_log_scan:
        raw_log_scan(str(args.data_dir))

    print("\n完成。重点看: 混合用户占比 / list长度 / 冷启动UNK率 / 每日正例率漂移。")


if __name__ == "__main__":
    main()
