import pandas as pd
from collections import Counter

D = "C:/Users/malla/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/"
rd = lambda p: pd.read_csv(D + p, sep="\t", dtype=str, keep_default_na=False, quoting=3)

gt = rd("train/train_ground_truth.tsv")
s1 = rd("train/train_source1.tsv")
s2 = rd("train/train_source2.tsv")
s3 = rd("train/train_source3.tsv")

lists = gt.matched_entity_ids.map(lambda x: [i for i in x.split(",") if i])
n = lists.map(len)
print("S1 rows:", len(s1), " GT rows:", len(gt))
print("singleton rate:", round((n == 0).mean(), 4))
print("matches per S1 distribution:\n", n.value_counts().sort_index().head(15))
n2 = lists.map(lambda l: sum(i.startswith("S2") for i in l))
n3 = lists.map(lambda l: sum(i.startswith("S3") for i in l))
print("mean S2 matches:", n2.mean(), " mean S3 matches:", n3.mean())

allm = [i for l in lists for i in l]
c = Counter(allm)
print("matched ids:", len(allm), " unique:", len(c), " ids in >1 S1 row:", sum(v > 1 for v in c.values()))
print("S2 records matched: %.3f of %d" % (sum(k.startswith("S2") for k in c) / len(s2), len(s2)))
print("S3 records matched: %.3f of %d" % (sum(k.startswith("S3") for k in c) / len(s3), len(s3)))

m = gt.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id")
m["n"] = n.values
print("\nby country (count, singleton rate, mean matches):")
print(m.groupby("country").agg(cnt=("n", "size"), single=("n", lambda x: (x == 0).mean()), mean=("n", "mean")))
for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(nm, "country:", df.country.value_counts().to_dict(),
          " empty addr:", round((df.business_address == "").mean(), 3),
          " empty name:", round((df.business_name == "").mean(), 3))

# show some matched examples
idx2 = pd.concat([s2, s3]).set_index("entity_id")
s1i = s1.set_index("entity_id")
for cn in ["US", "India"]:
    ex = m[(m.country == cn) & (m.n > 0)].sample(4, random_state=1)
    for _, r in ex.iterrows():
        a = s1i.loc[r.source1_entity_id]
        print(f"\n[{cn}] S1: {a.business_name} | {a.business_address}")
        for i in r.matched_entity_ids.split(","):
            b = idx2.loc[i]
            print(f"   {i[:2]}: {b.business_name} | {b.business_address}")
