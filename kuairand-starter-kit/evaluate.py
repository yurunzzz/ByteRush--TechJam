"""
KuaiRand-Pure 官方评测脚本 —— 口径全部写死在这里，不要改。

任务         : 用户内排序 (within-user ranking over logged impressions)
相关性标签   : long_view (原生列, 0/1)
指标         : GAUC, nDCG@5  (主分 = 两者的平均)
排序范围     : 每个用户只对其在评测集中的曝光排序, 不做全库检索
零正例用户   : nDCG 记为 0.0 并计入平均 (与 CWM 一致)
              GAUC 只统计 0 < 正例数 < 曝光数 的用户, 按正例数加权
nDCG gain    : (2^rel - 1), 二元标签下等价于 identity
数据划分     : train 20220408-20220421 / valid 20220422-20220428 / test 20220429-20220508
"""
import math, collections

def auc(labels, scores):
    """Mann-Whitney U，含并列修正，等价于 sklearn.metrics.roc_auc_score。"""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)

def ndcg_at_k(labels, k):
    """labels 已按预测分降序排列。"""
    disc = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate(user_ids, labels, scores, k=5):
    """返回 {'GAUC':…, 'nDCG@5':…, 'primary':…}。primary = 两者平均，用于排名。"""
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((s, y))
    gnum = gden = 0.0
    nd = []
    for u, lst in byu.items():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        npos = sum(labs)
        if 0 < npos < len(labs):
            gnum += npos * auc(labs, [s for s, _ in lst])
            gden += npos
        nd.append(ndcg_at_k(labs, k))
    gauc = gnum / gden if gden else 0.5
    ndcg = sum(nd) / len(nd) if nd else 0.0
    return {'GAUC': gauc, f'nDCG@{k}': ndcg, 'primary': (gauc + ndcg) / 2.0,
            'users': len(byu), 'rows': len(labels)}
