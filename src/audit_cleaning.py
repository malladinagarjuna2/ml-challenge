"""Verify the stated limitations of the cleaning, one by one, on real data."""
import os, re, sys, random, collections
from multiprocessing import Pool
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_record

D = "C:/Users/malla/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/"
rd = lambda f: pd.read_csv(D + f, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _norm(chunk):
    return [(k, normalize_record(*v)) for k, v in chunk]


def norm_all(recs):
    items = list(recs.items())
    chunks = [items[i:i + 5000] for i in range(0, len(items), 5000)]
    with Pool(20) as p:
        return dict(kv for part in p.map(_norm, chunks) for kv in part)


def jac(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0


if __name__ == "__main__":
    random.seed(0)
    gt = rd("train/train_ground_truth.tsv").sample(100000, random_state=1)
    need = {i for l in gt.matched_entity_ids for i in l.split(",") if i}
    s1ids = set(gt.source1_entity_id)
    recs = {}
    for f in ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv"]:
        df = rd(f)
        keep = s1ids if "source1" in f else need
        df = df[df.entity_id.isin(keep)]
        recs.update({r.entity_id: (r.business_name, r.business_address, r.country) for r in df.itertuples()})
    N = norm_all(recs)
    owner = {b: a for a, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in l.split(",") if b}
    pairs = [(a, b) for b, a in owner.items()]
    print(f"sample: {len(s1ids):,} S1 entities, {len(pairs):,} true pairs\n")

    # ---------------------------------------------------------------- claim 5: state disagreement
    print("=== CLAIM 5: states disagree in ~2.4% of true pairs; why? ===")
    mis = collections.Counter()
    both = 0
    for a, b in pairs:
        sa, sb = N[a]["state"], N[b]["state"]
        if sa and sb:
            both += 1
            if sa != sb:
                mis[(N[a]["country_n"], sa, sb)] += 1
    tot = sum(mis.values())
    print(f"disagree: {tot:,}/{both:,} = {tot / both:.2%}")
    for k, v in mis.most_common(12):
        print(f"   {k}: {v}")
    ex = [(a, b) for a, b in pairs if N[a]["state"] and N[b]["state"] and N[a]["state"] != N[b]["state"]][:6]
    for a, b in ex:
        print(f"   S1 {recs[a][1]!r}\n      -> {recs[b][1]!r}")

    # ---------------------------------------------------------------- claim 2: delhi / st louis
    print("\n=== CLAIM 2: stripping 'delhi' / 'st'->'street' — does it hurt? ===")
    for label, cond in [("Delhi (India)", lambda r: "delhi" in r[1].lower()),
                        ("'St ' + city-like (US)", lambda r: r[2] == "US" and re.search(r"\b(st|saint)\.? (louis|paul|george|petersburg|charles|augustine|cloud)\b", r[1].lower()))]:
        sel = [(a, b) for a, b in pairs if cond(recs[a])]
        emp = sum(1 for a, _ in sel if not N[a]["addr"]) / max(len(sel), 1)
        sim = sum(fuzz.token_set_ratio(N[a]["addr"], N[b]["addr"]) for a, b in sel) / max(len(sel), 1)
        allsim = sum(fuzz.token_set_ratio(N[a]["addr"], N[b]["addr"]) for a, b in pairs) / len(pairs)
        print(f"   {label}: {len(sel):,} true pairs, cleaned addr empty {emp:.2%}, "
              f"addr sim {sim:.1f} (all pairs {allsim:.1f})")
    delhi = [a for a in s1ids if "delhi" in recs[a][1].lower()][:5]
    for a in delhi:
        print(f"   {recs[a][1]!r} -> {N[a]['addr']!r}")

    # ---------------------------------------------------------------- claim 3: learn substitutions
    print("\n=== CLAIM 3: which token substitutions do true pairs contain that the hand lists miss? ===")
    subs = {"addr": collections.Counter(), "name": collections.Counter()}
    for a, b in pairs:
        for fld, key in [("addr", "addr"), ("name", "name")]:
            ta, tb = N[a][key].split(), N[b][key].split()
            oa, ob = [t for t in ta if t not in tb], [t for t in tb if t not in ta]
            if len(oa) == 1 and len(ob) == 1 and not oa[0].isdigit() and not ob[0].isdigit():
                subs[fld][(oa[0], ob[0])] += 1
    for fld in subs:
        print(f"   top {fld} substitutions (S1 token -> S2/S3 token):")
        print("     ", ", ".join(f"{x}->{y}:{c}" for (x, y), c in subs[fld].most_common(30)))

    # ---------------------------------------------------------------- claim 4: hard negatives
    print("\n=== CLAIM 4: hard negatives (look-alikes) vs true pairs ===")
    s23 = [b for b in owner]
    by_addr = collections.defaultdict(list)   # same house number + same street words -> shared address
    by_name = collections.defaultdict(list)   # same first strict-name token -> similar name
    for b in s23:
        n = N[b]
        if n["nums"] and n["addr_street"]:
            by_addr[(n["country_n"], n["nums"].split()[0], " ".join(n["addr_street"].split()[:2]))].append(b)
        if n["name_strict"]:
            by_name[(n["country_n"], n["name_strict"].split()[0])].append(b)
    hn_addr, hn_name = [], []
    for a in s1ids:
        n = N[a]
        if n["nums"] and n["addr_street"]:
            for b in by_addr[(n["country_n"], n["nums"].split()[0], " ".join(n["addr_street"].split()[:2]))]:
                if owner[b] != a:
                    hn_addr.append((a, b))
        if n["name_strict"]:
            for b in by_name[(n["country_n"], n["name_strict"].split()[0])][:20]:
                if owner[b] != a:
                    hn_name.append((a, b))
    random.shuffle(hn_name)
    hn_name = hn_name[:50000]

    def prof(pp):
        k = len(pp) or 1
        nm = sum(fuzz.token_set_ratio(N[a]["name_strict"], N[b]["name_strict"]) for a, b in pp) / k
        ad = sum(fuzz.token_set_ratio(N[a]["addr"], N[b]["addr"]) for a, b in pp) / k
        nb = [(a, b) for a, b in pp if N[a]["nums"] and N[b]["nums"]]
        no = sum(bool(set(N[a]["nums"].split()) & set(N[b]["nums"].split())) for a, b in nb) / (len(nb) or 1)
        return len(pp), nm, ad, no
    print(f"   {'set':34s} {'n':>8s} {'name sim':>9s} {'addr sim':>9s} {'nums overlap':>13s}")
    for lbl, pp in [("true pairs", pairs), ("hard neg: same address, other biz", hn_addr),
                    ("hard neg: same first name word", hn_name)]:
        n_, nm, ad, no = prof(pp)
        print(f"   {lbl:34s} {n_:8,d} {nm:9.1f} {ad:9.1f} {no:13.1%}")
    for a, b in hn_addr[:5]:
        print(f"   S1 {recs[a][0]} | {recs[a][1]}\n      ≠ {recs[b][0]} | {recs[b][1]}")

    # ---------------------------------------------------------------- claim 1: France, label-free
    print("\n=== CLAIM 1: France, label-free checks on test data ===")
    fr = {}
    for f in ["test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv"]:
        df = rd(f)
        df = df[df.country == "France"].sample(30000, random_state=0)
        fr.update({r.entity_id: (r.business_name, r.business_address, r.country) for r in df.itertuples()})
    FN = norm_all(fr)
    for src in ("S1", "S2", "S3"):
        ids = [k for k in FN if k.startswith(src)]
        g = lambda f: sum(bool(FN[k][f]) for k in ids) / len(ids)
        print(f"   {src}: nums {g('nums'):.1%}  postcode {g('postcode'):.1%}  region {g('region'):.1%}  "
              f"legal {g('legal'):.1%}  addr empty {g('addr_missing'):.1%}")
    # vocabulary audit: frequent S2/S3 address tokens that never/rarely occur in S1 = unhandled variants
    v1, v23 = collections.Counter(), collections.Counter()
    for k, n in FN.items():
        (v1 if k.startswith("S1") else v23).update(n["addr"].split())
    odd = [(t, c) for t, c in v23.most_common(3000) if v1[t] < c * 0.02 and not t.isdigit() and len(t) < 12]
    print("   frequent S2/S3 address tokens rare in S1 (unhandled variants?):")
    print("     ", ", ".join(f"{t}:{c}" for t, c in odd[:40]))
    n1, n23 = collections.Counter(), collections.Counter()
    for k, n in FN.items():
        (n1 if k.startswith("S1") else n23).update(n["name"].split())
    odd = [(t, c) for t, c in n23.most_common(3000) if n1[t] < c * 0.02 and not t.isdigit()]
    print("   frequent S2/S3 name tokens rare in S1:")
    print("     ", ", ".join(f"{t}:{c}" for t, c in odd[:40]))
