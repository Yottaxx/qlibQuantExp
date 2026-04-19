import qlib, time
import pandas as pd
from qlib.data import D
import work_flow
log = open(r'next_step\hierarchical_state_field\experiment\runs\shared_obs_benchmark_field_active_block_manual\coverage_probe.log','a',encoding='utf-8', buffering=1)
def p(*args): print(*args, file=log, flush=True)
qlib.init(provider_uri=work_flow.provider_uri, region=work_flow.REG_CN)
p('init done')
spans=D.list_instruments(D.instruments('all'), start_time='2008-01-01', end_time='2022-12-31', freq='day', as_list=False)
p('spans', len(spans))
def active_for(ds):
    dt=pd.Timestamp(ds)
    out=[]
    for inst,sps in spans.items():
        for sp in sps or []:
            if pd.Timestamp(sp[0]) <= dt <= pd.Timestamp(sp[1]): out.append(str(inst)); break
    return out
for ds in ['2008-01-02']:
    active=active_for(ds)
    p('DATE', ds, 'active', len(active))
    got=set(); rows=0; nonna=0; t0=time.time()
    for i in range(0,len(active),200):
        sub=active[i:i+200]
        p('chunk', i, 'n', len(sub))
        df=D.features(sub, ['$close','$volume','$amount'], start_time=ds, end_time=ds, freq='day')
        p('chunk_done', i, 'empty', (df is None or df.empty), 'sec', round(time.time()-t0,2))
        if df is None or df.empty: continue
        rows += len(df); nonna += int(df.notna().all(axis=1).sum())
        idx=df.index; names=list(idx.names or []); inst_level=names.index('instrument') if 'instrument' in names else 0
        got.update(map(str, idx.get_level_values(inst_level)))
    p('RESULT', ds, rows, nonna, len(got), len(got)/len(active) if active else None, 'sec', round(time.time()-t0,2))
