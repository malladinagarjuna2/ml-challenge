import pandas as pd, numpy as np, csv, re, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
rng=np.random.default_rng(0)
gt=pd.read_csv("C:/Users/malla/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",sep="\t",dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE)
COLS=['entity_id','business_name','business_address','country_n','n_core','n_legal','n_alias','a_street','a_nums','a_state','a_zip','n_nospace']
L_full=pq.read_table('cache/v2/train_s1.parquet',columns=['entity_id','country_n','n_core','a_state','a_street']).to_pandas()
# name-duplicate stats on full S1
key=L_full.country_n+'|'+L_full.a_state+'|'+L_full.n_core
dupn=key.map(key.value_counts())
samp=gt.sample(150000,random_state=0)
pairs=[(s,r) for s,m in zip(samp.source1_entity_id,samp.matched_entity_ids) for r in m.split(',') if r]
P=pd.DataFrame(pairs,columns=['l','r'])
def load(path,ids):
    t=pq.read_table(path,columns=COLS); t=t.filter(pc.is_in(t['entity_id'],value_set=pa.array(list(ids)))); return t.to_pandas()
L=load('cache/v2/train_s1.parquet',set(P.l)).set_index('entity_id')
L['dupn']=dupn.values[pd.Index(L_full.entity_id).get_indexer(L.index)]
R=pd.concat([load(f'cache/v2/train_s{s}.parquet',set(P.r)) for s in (2,3)]).set_index('entity_id')
A=L.loc[P.l].reset_index(drop=True).add_prefix('l_'); B=R.loc[P.r].reset_index(drop=True).add_prefix('r_')
D=pd.concat([P,A,B],axis=1)
W=dict(workers=-1,dtype=np.float32)
D['nsim']=cpdist(D.l_n_core.tolist(),D.r_n_core.tolist(),scorer=fuzz.token_set_ratio,**W)
D['asim']=cpdist(D.l_a_street.tolist(),D.r_a_street.tolist(),scorer=fuzz.token_set_ratio,**W)
D['shared']=[bool(set(a.split())&set(b.split())) for a,b in zip(D.l_n_core,D.r_n_core)]
INDIC=re.compile("[\u0900-\u0dff]"); WEB=re.compile(r"\.(com|net|org|in|co|biz)\b|www\.",re.I)
def initials(s): return ''.join(w[0] for w in s.split() if w)
def cat(x):
    ln,rn,lraw,rraw=x.l_n_core,x.r_n_core,x.l_business_name,x.r_business_name
    if rn=='' : return 'R name empty'
    if x.shared: return None
    if WEB.search(rraw) or WEB.search(lraw): return 'website as name'
    if INDIC.search(rraw): return 'Indic script (after translit)'
    if x.shared: return None
    ln_ns,rn_ns=ln.replace(' ',''),rn.replace(' ','')
    if len(rn_ns)>=4 and (rn_ns in ln_ns or ln_ns in rn_ns): return 'spacing / glued words'
    if (len(rn_ns)<=5 and rn_ns==initials(ln)) or (len(ln_ns)<=5 and ln_ns==initials(rn)): return 'acronym / initials'
    if fuzz.ratio(ln,rn)>=70: return 'typos in every word'
    if x.r_n_alias not in ('',None) : return 'DBA / alias name'
    return 'unrelated name (generated / trade name)'
D['cat']=[cat(x) for x in D.itertuples()]
hard_name=D.cat.notna()
tot=len(D)
print(f"true pairs sampled: {tot:,}  | name gives no evidence: {hard_name.mean():.3f}")
t=D[hard_name].groupby('cat').agg(n=('l','size'),addr_ok=('asim',lambda s:(s>=80).mean()),addr_empty=('r_a_street',lambda s:(s=='').mean()))
t['share_of_all_true']=t.n/tot; print(t.sort_values('n',ascending=False).round(3).to_string())
# address-side hard cases among pairs where name is fine
ok=~hard_name
print("\n-- name fine, address weak --")
print("R address empty:",round(((D.r_a_street=='')&ok).mean(),4))
print("L address empty:",round(((D.l_a_street=='')&ok).mean(),4))
m=ok&(D.r_a_street!='')&(D.l_a_street!='')&(D.asim<50); print("address very different (<50):",round(m.mean(),4))
ln1=D.l_a_nums.str.split().str[0].fillna(''); rn1=D.r_a_nums.str.split().str[0].fillna('')
print("house number differs:",round((ok&(ln1!='')&(rn1!='')&(ln1!=rn1)).mean(),4))
print("state differs:",round(((D.l_a_state!='')&(D.r_a_state!='')&(D.l_a_state!=D.r_a_state)).mean(),4))
print("\n-- both weak (name no evidence AND address <60 or empty) --", round((hard_name&((D.asim<60)|(D.r_a_street=='')|(D.l_a_street==''))).mean(),4))
print("\n-- precision side: S1 whose exact name+state is shared by other S1 records --")
print("share of S1 (all train):",round((dupn>1).mean(),4),"| share of true pairs:",round((D.l_dupn>1).mean(),4))
with open('runs/hard_examples.txt','w',encoding='utf-8') as f:
    for c,g in D[hard_name].groupby('cat'):
        f.write(f"\n=== {c}\n")
        for x in g.sample(min(6,len(g)),random_state=1).itertuples():
            f.write(f"  S1: {x.l_business_name} | {x.l_business_address}\n  R : {x.r_business_name} | {x.r_business_address}   (addr sim {x.asim:.0f})\n")
    f.write("\n=== name fine, address very different\n")
    for x in D[m].sample(6,random_state=1).itertuples():
        f.write(f"  S1: {x.l_business_name} | {x.l_business_address}\n  R : {x.r_business_name} | {x.r_business_address}\n")
    f.write("\n=== same name+state shared by several S1\n")
    for x in D[D.l_dupn>2].sample(6,random_state=1).itertuples():
        f.write(f"  S1: {x.l_business_name} | {x.l_business_address}  (x{x.l_dupn})\n  R : {x.r_business_name} | {x.r_business_address}\n")
