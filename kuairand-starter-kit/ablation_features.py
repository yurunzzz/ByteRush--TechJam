"""把 CWM 的 13 个特征域接进来，验证「用户侧特征在 FM 里是否有用」。"""
import csv, os, collections, statistics
import numpy as np
from evaluate import evaluate
import baseline as B

import sys
D = sys.argv[1] if len(sys.argv) > 1 else './KuaiRand-Pure/data'
SPLITS={'train':(20220408,20220421),'valid':(20220422,20220428),'test':(20220429,20220508)}

# CWM 的 13 个域
USER_FE=['follow_user_num_range','register_days_range','fans_user_num_range',
         'friend_user_num_range','user_active_degree']
VID_FE=['author_id','music_id','video_type','upload_type']

u_ext={}
with open(f'{D}/user_features_pure.csv') as fh:
    for r in csv.DictReader(fh): u_ext[r['user_id']]=[r[k] for k in USER_FE]
v_ext={}
with open(f'{D}/video_features_basic_pure.csv') as fh:
    for r in csv.DictReader(fh):
        v_ext[r['video_id']]=[r[k] for k in VID_FE[:1]+VID_FE[1:]]

rows=[]
for f in ('log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv'):
    with open(f'{D}/{f}') as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r['date']), r['user_id'], r['video_id'], r['tab'],
                         float(r['duration_ms']), 1 if r['long_view']!='0' else 0))
splits={n:[x for x in rows if lo<=x[0]<=hi] for n,(lo,hi) in SPLITS.items()}
print({k:len(v) for k,v in splits.items()})

UNKU=['UNK']*len(USER_FE); UNKV=['UNK']*len(VID_FE)
def build(mode):
    """mode: 'base'=5域(现kit) / 'item'=只加物品侧 / 'cwm13'=CWM全13域"""
    edges=np.quantile([x[4] for x in splits['train']], np.linspace(0,1,11)[1:-1])
    def raw(x):
        ue=u_ext.get(x[1],UNKU); ve=v_ext.get(x[2],UNKV)
        f=[x[1], x[2], ve[0], x[3], str(int(np.searchsorted(edges,x[4])))]   # 5 域基线
        if mode in ('item','cwm13'): f += ve[1:]                              # +music/type/upload
        if mode=='cwm13':            f += ue                                  # +6 用户侧
        return f
    n=len(raw(splits['train'][0]))
    vocabs=[dict() for _ in range(n)]
    for x in splits['train']:
        for i,v in enumerate(raw(x)):
            if v not in vocabs[i]: vocabs[i][v]=len(vocabs[i])
    unk=[len(v) for v in vocabs]; dims=[len(v)+1 for v in vocabs]
    off=np.cumsum([0]+dims[:-1]).astype(np.int32)
    enc={}
    for name,rws in splits.items():
        X=np.empty((len(rws),n),dtype=np.int32); y=np.empty(len(rws),dtype=np.float32); us=[]
        for j,x in enumerate(rws):
            for i,v in enumerate(raw(x)): X[j,i]=vocabs[i].get(v,unk[i])+off[i]
            y[j]=x[5]; us.append(x[1])
        enc[name]=(X,y,us)
    return enc, int(sum(dims)), n

for mode,desc in [('base','5 域（当前 kit）'),('item','+4 物品侧 = 9 域'),('cwm13','CWM 全 13 域')]:
    enc,dim,nf=build(mode)
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; Xte,yte,ute=enc['test']
    scores=[]
    for seed in range(3):
        m=B.FM(dim,k=16,lr=0.001,seed=seed); rng=np.random.default_rng(seed)
        best=-1; bs=8192; bad=0; state=None
        for ep in range(40):
            idx=rng.permutation(len(ytr))
            for i in range(0,len(idx),bs): m.step(Xtr[idx[i:i+bs]], ytr[idx[i:i+bs]])
            p=evaluate(uva,yva,m.predict(Xva))['primary']
            if p>best+1e-5: best=p; bad=0; state=(m.V.copy(),m.W.copy(),np.float32(m.b))
            else:
                bad+=1
                if bad>=4: break
        m.V,m.W,m.b=state
        scores.append(evaluate(ute,yte,m.predict(Xte)))
    g=statistics.mean(s['GAUC'] for s in scores); n5=statistics.mean(s['nDCG@5'] for s in scores)
    pr=statistics.mean(s['primary'] for s in scores); sd=statistics.pstdev([s['primary'] for s in scores])
    print(f"{desc:20s} ({nf:2d}域) | test GAUC {g:.4f} | nDCG@5 {n5:.4f} | primary {pr:.4f} ± {sd:.4f}")
