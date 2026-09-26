"""Sanity-check normalization on ground-truth pairs: does cleaning bring true matches closer?"""
import os, re, sys, random, collections
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_record

D = "C:/Users/malla/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/"
rd = lambda f: pd.read_csv(D + f, sep="\t", dtype=str, keep_default_na=False, quoting=3)

random.seed(0)
gt = rd("train_ground_truth.tsv").sample(20000, random_state=0)
need = {i for l in gt.matched_entity_ids for i in l.split(",") if i}
recs = {}
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    df = rd(f)
    if "source1" in f:
        df = df[df.entity_id.isin(set(gt.source1_entity_id))]
    else:
        df = df[df.entity_id.isin(need)]
    recs.update({r.entity_id: (r.business_name, r.business_address, r.country) for r in df.itertuples()})
norm = {k: normalize_record(*v) for k, v in recs.items()}

pairs = [(a, b) for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in l.split(",") if b]
# negatives: S1 paired with a matched record of a *different* S1 in the same country and same state
by_state = collections.defaultdict(list)
for a, b in pairs:
    by_state[(norm[b]["country_n"], norm[b]["state"])].append((a, b))
negs = []
for a, b in pairs[:20000]:
    pool = by_state[(norm[a]["country_n"], norm[a]["state"])]
    x = random.choice(pool)
    if x[0] != a:
        negs.append((a, x[1]))


def stats(pp):
    out = collections.defaultdict(float)
    for a, b in pp:
        ra, rb = recs[a], recs[b]
        na, nb = norm[a], norm[b]
        out["name raw tsr"] += fuzz.token_set_ratio(ra[0].lower(), rb[0].lower())
        out["name clean tsr"] += max([fuzz.token_set_ratio(na["name_strict"], nb["name_strict"])] +
                                     [fuzz.token_set_ratio(na["name_strict"], v) for v in nb["name_alts"].split("|") if v])
        out["name_strict equal"] += na["name_strict"] == nb["name_strict"]
        out["addr raw tsr"] += fuzz.token_set_ratio(ra[1].lower(), rb[1].lower())
        out["addr clean tsr"] += fuzz.token_set_ratio(na["addr"], nb["addr"])
        both = na["nums"] and nb["nums"]
        out["_nums both"] += bool(both)
        out["nums overlap|both"] += bool(both and set(na["nums"].split()) & set(nb["nums"].split()))
        out["_state both"] += bool(na["state"] and nb["state"])
        out["state equal|both"] += bool(na["state"] and na["state"] == nb["state"])
        out["b addr missing"] += nb["addr_missing"]
    n = len(pp)
    res = {k: v / n for k, v in out.items() if not k.startswith("_")}
    res["nums overlap|both"] = out["nums overlap|both"] / max(out["_nums both"], 1)
    res["state equal|both"] = out["state equal|both"] / max(out["_state both"], 1)
    return res


P, N = stats(pairs), stats(negs)
print(f"{'metric':22s} {'match':>8s} {'non-match':>10s}   (n={len(pairs)} / {len(negs)})")
for k in P:
    print(f"{k:22s} {P[k]:8.3f} {N[k]:10.3f}")

# which transliterated tokens still survive as name words (candidates for the legal/generic maps)?
tok = collections.Counter()
for k, (n, a, c) in recs.items():
    if re.search(r"[ऀ-෿]", n):
        tok.update(norm[k]["name_strict"].split())
print("\nmost common tokens in transliterated names:", tok.most_common(40))

# worst true pairs after cleaning, to find what cleaning still misses
worst = sorted(pairs, key=lambda p: fuzz.token_set_ratio(norm[p[0]]["name_strict"], norm[p[1]]["name_strict"]))[:25]
print("\nhardest true pairs (name):")
for a, b in worst:
    print(f"  {recs[a][0]!r:45s} -> {recs[b][0]!r:45s} | clean: {norm[a]['name_strict']!r} vs {norm[b]['name_strict']!r}")
