"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits, feature_state=None, return_state=False):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    默认返回 (X, y, users) per split 和 field_dims。
    return_state=True 时额外返回只在 train 上拟合的编码状态；传入
    feature_state 可将同一状态原样用于 valid/test。"""
    if feature_state is None:
        tr = splits['train']
        edges = _bucket_edges([x[5] for x in tr])
    else:
        if feature_state.get('fields') != FIELDS:
            raise ValueError('feature_state fields 与当前 FIELDS 不一致')
        edges = np.asarray(feature_state['edges'], dtype=np.float64)

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    if feature_state is None:
        vocabs = [dict() for _ in FIELDS]
        for x in tr:
            for i, v in enumerate(raw(x)):
                if v not in vocabs[i]:
                    vocabs[i][v] = len(vocabs[i])
        unk = [len(v) for v in vocabs]             # 每个域末尾留一个 UNK 槽
        field_dims = [len(v) + 1 for v in vocabs]
        offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
        feature_state = {
            'schema_version': 1,
            'fields': list(FIELDS),
            'edges': edges.tolist(),
            'vocabs': vocabs,
            'unk': unk,
            'field_dims': field_dims,
            'offsets': offsets.tolist(),
        }
    else:
        vocabs = feature_state['vocabs']
        unk = [int(v) for v in feature_state['unk']]
        field_dims = [int(v) for v in feature_state['field_dims']]
        offsets = np.asarray(feature_state['offsets'], dtype=np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    result = (enc, int(sum(field_dims)))
    if return_state:
        return result + (feature_state,)
    return result
