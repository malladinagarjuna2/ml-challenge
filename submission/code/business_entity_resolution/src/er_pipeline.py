"""Amazon ML Challenge 2026 - Business Entity Resolution: full pipeline (exported from er_pipeline.ipynb).

Run:  ER_ROOT=<student_resource dir>  ER_WORK=<work dir>  python er_pipeline.py
"""

# # Amazon ML Challenge 2026 — Business Entity Resolution (pipeline notebook)
# 
# Stages: **0** config · **1** load · **2** preprocessing (fixed + tested + quality report) · **3** sample ·
# **4** embeddings (optional) · **5** blocking · **6** features · **7** metric · **8** stage-1 LightGBM ·
# **9** stage-2 LightGBM · **10** decision layer (macro F0.5) · **11** write + validate
# 
# **Preprocessing fixes vs the original Colab version**
# 1. States are read *before* abbreviations are expanded, so `FL / CT / MT / NE` stay states (they used to become floor / court / mount / northeast).
# 2. A state is only taken from a whole address component or the end of a number-free component, checked **last component first**, and only with that country's table. So `rue de la paix` no longer gives state `la`, `Washington, DC` gives `dc`, and `Kansas City, MO` gives `mo`.
# 3. Abbreviations are country-specific. `saint` and `st` are consistent (US: street, France: saint), and French rules (`r`→rue, `crs`, `psg`, `N°`) don't touch US addresses.
# 4. Indian state names in native scripts (`महाराष्ट्र` → `mharastr`) are **learned from the training ground truth** and mapped to state codes.
# 5. `N°` no longer becomes `ndeg`, `+` becomes "and" like `&`, `NULL` becomes empty, and renamed Indian cities (Bombay/Mumbai, Gurgaon/Gurugram…) are unified.
# 6. The postcode is extracted per country (US/FR 5 digits, IN 6 digits) and never from the leading house number. Previously any number with 5+ digits was a "zip".
# 7. Alias markers (dba / formerly / f/k/a / aka / t/a) and honorifics (Shri, Smt, M/s…) are removed from the core name, and a flag is kept.
# 8. `prepare()` runs on all CPU cores and caches its output to parquet. Number and word lists are stored as strings, not Python lists (memory).
# 
# **Scale fixes:** blocking runs per country with `sparse_dot_topn` (the old dense top-k needed about 40 GB per chunk), views are merged with numpy instead of `pivot_table`, and the F0.5 scoring used in the threshold search is vectorised.

# ============================================================ 0. CONFIG
import os, re, csv, gc, sys, time, math, json, unicodedata, warnings, subprocess
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from joblib import Parallel, delayed
from sparse_dot_topn import sp_matmul_topn
warnings.filterwarnings("ignore")

ROOT = os.environ.get("ER_ROOT", "student_resource")
DATA_DIR = f"{ROOT}/dataset"
PROJ = os.environ.get("ER_WORK", ".")
CACHE_DIR = f"{PROJ}/cache"

MODE = "FULL"           # "DEV" = quick end-to-end check on a sample | "FULL" = real submission
OUT_DIR = f"{PROJ}/output" if MODE == "FULL" else f"{PROJ}/output_dev"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

N_JOBS = max(1, (os.cpu_count() or 2) - 2)
SEED = 42
N_FOLDS = 5
K_TFIDF = 15            # neighbours per view per source (S2 and S3 separately)
K_EMB = 15
USE_EMBEDDINGS = False  # GPU embeddings: next experiment (22M texts ~1-1.5 h, ~17 GB -> needs disk memmap)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0, 118M params
MAX_CAND_PER_S1 = 40   # 5b: RRF-ranked top-40 keeps 97.7% of true pairs (all candidates: 98.3%)
FORCE_PREP = False
RUN_BLOCK_EXPERIMENT = False   # 5b: 20k-S1 blocking study (~16 min), not needed for the outputs      # re-run normalisation even if the parquet cache exists

DEV_TRAIN_S1, DEV_TEST_S1 = 50_000, 20_000   # DEV samples S1 only; S2/S3 are always complete
FULL_TRAIN_S1 = 200_000  # FULL: train on a sample of S1 (blocked against ALL of train S2/S3)

np.random.seed(SEED)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log("mode", MODE, "| cpu jobs", N_JOBS)

# ============================================================ 1. LOAD
def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")

def load_split(split):
    d = {}
    for s in (1, 2, 3):
        df = read_tsv(f"{DATA_DIR}/{split}/{split}_source{s}.tsv")
        df.columns = [c.strip() for c in df.columns]
        for c in ["entity_id", "business_name", "business_address", "country"]:
            if c not in df.columns: df[c] = ""
        df["entity_id"] = df["entity_id"].str.strip()
        d[s] = df.reset_index(drop=True)
    return d

def dedup_ids(d, split):
    for s in d:
        nd = d[s]["entity_id"].duplicated().sum()
        ne = (d[s]["entity_id"] == "").sum()
        if nd or ne: log(f"!! {split} source{s}: {nd} duplicate ids, {ne} empty ids -> dropped")
        d[s] = d[s][d[s]["entity_id"] != ""].drop_duplicates("entity_id").reset_index(drop=True)
    return d

CACHE_FILES = {(sp, s): f"{CACHE_DIR}/{sp}_s{s}.parquet" for sp in ("train", "test") for s in (1, 2, 3)}
HAVE_CACHE = (not FORCE_PREP) and all(os.path.exists(p) for p in CACHE_FILES.values())

gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
gt.columns = [c.strip() for c in gt.columns]
GT = {r.source1_entity_id.strip(): set(x.strip() for x in r.matched_entity_ids.split(",") if x.strip())
      for r in gt.itertuples()}

if HAVE_CACHE:
    log("normalised parquet cache found -> raw TSV load skipped")
else:
    tr = dedup_ids(load_split("train"), "train")
    te = dedup_ids(load_split("test"), "test")
    for k, d in (("train", tr), ("test", te)):
        log(k, {s: len(d[s]) for s in d})
    log("country counts train S1:", tr[1]["country"].value_counts().to_dict())
    log("country counts test  S1:", te[1]["country"].value_counts().to_dict())

sizes = np.array([len(v) for v in GT.values()])
log(f"GT: {len(GT):,} S1 rows | singleton rate={np.mean(sizes==0):.3f} | mean matches={sizes.mean():.2f} | max={sizes.max()}")
owner_cnt = Counter(x for v in GT.values() for x in v)
ONE_TO_ONE = sum(1 for c in owner_cnt.values() if c > 1) == 0
log(f"ONE_TO_ONE (no S2/S3 id matched to >1 S1) = {ONE_TO_ONE}")

# ## 2. Preprocessing
# **2a** defines the normalisers. **2b** checks them on tricky real cases with asserts, so a failing assert means a preprocessing bug.
# **2c** learns native-script state names from the training labels. **2d** normalises all 23.4M rows (in parallel, cached).
# **2e** is a quality report on the full output.

# ============================================================ 2a. NORMALISATION
US_STATES = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
 "connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl","georgia":"ga","hawaii":"hi",
 "idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la",
 "maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms",
 "missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj",
 "new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
 "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd",
 "tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa",
 "west virginia":"wv","wisconsin":"wi","wyoming":"wy"}
IN_STATES = {"andhra pradesh":"ap","arunachal pradesh":"ar","assam":"as","bihar":"br","chhattisgarh":"cg",
 "goa":"ga","gujarat":"gj","haryana":"hr","himachal pradesh":"hp","jharkhand":"jh","karnataka":"ka",
 "kerala":"kl","madhya pradesh":"mp","maharashtra":"mh","manipur":"mn","meghalaya":"ml","mizoram":"mz",
 "nagaland":"nl","odisha":"od","orissa":"od","punjab":"pb","rajasthan":"rj","sikkim":"sk","tamil nadu":"tn",
 "tamilnadu":"tn","telangana":"tg","tripura":"tr","uttar pradesh":"up","uttarakhand":"uk","west bengal":"wb",
 "delhi":"dl","new delhi":"dl","nct of delhi":"dl","jammu and kashmir":"jk","puducherry":"py",
 "pondicherry":"py","chandigarh":"ch","keralam":"kl"}
IN_CODE_ALIAS = {"ts":"tg","tl":"tg","or":"od","ut":"uk"}          # alternative state codes seen in India
STATE_TABLES = {  # country -> (full name -> code, accepted codes, code aliases)
    "us": (US_STATES, set(US_STATES.values()), {}),
    "india": (IN_STATES, set(IN_STATES.values()) | set(IN_CODE_ALIAS), IN_CODE_ALIAS),
}   # France / unseen countries: no state table -> state stays "" (unknown), never guessed
ZIP_LEN = {"us": 5, "france": 5, "india": 6}

EN_STREET = {"st":"street","str":"street","saint":"street","rd":"road","ave":"avenue","av":"avenue","avn":"avenue",
 "blvd":"boulevard","bd":"boulevard","dr":"drive","drv":"drive","ln":"lane","ct":"court","crt":"court","pl":"place",
 "hwy":"highway","pkwy":"parkway","cir":"circle","trl":"trail","sq":"square","ter":"terrace","terr":"terrace",
 "cres":"crescent","mt":"mount","ft":"fort","pt":"point","jn":"junction","jct":"junction","expy":"expressway",
 "fwy":"freeway","cv":"cove","xing":"crossing","rdg":"ridge","crk":"creek","hts":"heights","mtn":"mountain",
 "vly":"valley","hl":"hill","hls":"hills","grv":"grove","lk":"lake","spg":"spring","sta":"station",
 "tpke":"turnpike","ctr":"center","fls":"falls","gdns":"gardens","hbr":"harbor","holw":"hollow","lndg":"landing",
 "mdw":"meadow","pnes":"pines","rnch":"ranch","shr":"shore","vw":"view","vlg":"village","aly":"alley",
 "n":"north","s":"south","e":"east","w":"west","ne":"northeast","nw":"northwest","se":"southeast","sw":"southwest",
 "nr":"near","opp":"opposite","bldg":"building","blk":"block","fl":"floor","flr":"floor","apt":"unit",
 "apartment":"unit","ste":"unit","suite":"unit","rte":"route","po":"post","hno":"house",
 "marg":"road","salai":"road","sadak":"road","sec":"sector","mkt":"market","extn":"extension",
 "first":"1","second":"2","third":"3","fourth":"4","fifth":"5"}
FR_STREET = {"r":"rue","av":"avenue","ave":"avenue","avn":"avenue","bd":"boulevard","bld":"boulevard",
 "blvd":"boulevard","imp":"impasse","ch":"chemin","che":"chemin","chem":"chemin","rte":"route","all":"allee",
 "alle":"allee","alee":"allee","pl":"place","fbg":"faubourg","q":"quai","qu":"quai","crs":"cours",
 "psg":"passage","pass":"passage","app":"unit","appt":"unit","apt":"unit","st":"saint","ste":"saint",
 "sainte":"saint","bat":"batiment","res":"residence","sq":"square","prom":"promenade","lot":"lotissement"}
STREET_BY_COUNTRY = {"us": EN_STREET, "india": EN_STREET, "france": FR_STREET}
CITY_ALIAS = {"bombay":"mumbai","calcutta":"kolkata","madras":"chennai","bangalore":"bengaluru",
 "bengalooru":"bengaluru","gurgaon":"gurugram","trivandrum":"thiruvananthapuram","mysore":"mysuru",
 "poona":"pune","baroda":"vadodara","cochin":"kochi","pondicherry":"puducherry","benares":"varanasi",
 "banaras":"varanasi","allahabad":"prayagraj","mangalore":"mangaluru","belgaum":"belagavi","hubli":"hubballi",
 "vizag":"visakhapatnam","calicut":"kozhikode","trichy":"tiruchirappalli","tuticorin":"thoothukudi","simla":"shimla"}
ADDR_DROP = {"unit","floor","no","number","plot","flat","house","h","door","shop","shp","po","box","city","county",
 "town","village","of","the","near","opposite","building","block","ground","and","district","dist","taluk",
 "tehsil","post","cedex","bis","ter","india","usa","us","united","states","france","cdp"}

LEGAL = {"incorporated":"inc","inc":"inc","lnc":"inc","corporation":"corp","corp":"corp","corpn":"corp",
 "company":"co","co":"co","limited":"ltd","ltd":"ltd","private":"pvt","pvt":"pvt","pvtltd":"pvt ltd","llc":"llc",
 "lc":"llc","llp":"llp","lp":"lp","pc":"pc","pllc":"pllc","pa":"pa","plc":"plc","group":"group",
 "associates":"assoc","assoc":"assoc","services":"svc","service":"svc","svcs":"svc","gmbh":"gmbh","sa":"sa",
 "sas":"sas","sarl":"sarl","eurl":"eurl","sasu":"sasu","sci":"sci","snc":"snc","holding":"holding",
 "holdings":"holding","international":"intl","intl":"intl","enterprises":"ent","enterprise":"ent",
 "industries":"ind","technologies":"tech","technology":"tech","solutions":"soln","solution":"soln","dba":"dba"}
LEGAL_FORMS = {"inc","corp","co","ltd","pvt","llc","llp","lp","pc","pllc","pa","plc","gmbh","sa","sas","sarl",
               "eurl","sasu","sci","snc"}
HONORIFIC = {"m/s","ms","mr","mrs","smt","shri","sri","shree","dr"}
ALIAS_RE = re.compile(r"\b(doing business as|d/b/a|dba|formerly known as|formerly|f/k/a|fka|"
                      r"trading as|t/a|a/k/a|aka)\b")
NAME_STOP = set(LEGAL_FORMS) | HONORIFIC | {"the","and","of","dba","a","an","le","la","les","de","du","des","et"}
HOMO = str.maketrans({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t","8":"b","@":"a","$":"s"})
ORD_RE = re.compile(r"^\d+(st|nd|rd|th)$")
NUMSIGN_RE = re.compile(r"\b[nN]\s*[°º]|[°º]")                  # French 'N° 12' -> 'no 12' (not 'ndeg')
JUNK_RE = re.compile(r"\b(null|none|nan|n/a)\b")
NATIVE_RE = re.compile(r"[\u0900-\u0dff]")                     # Indic scripts


def phon(s):
    """crude phonetic key: kills transliteration variance (Raebareli ~ Raibareilly, Hindi/Tamil unidecode)."""
    s = re.sub(r"([a-z])\1+", r"\1", s)
    for a, b in (("ph","f"),("bh","b"),("kh","k"),("gh","g"),("dh","d"),("th","t"),("sh","s"),("ch","c"),
                 ("aa","a"),("ee","i"),("ii","i"),("oo","u"),("uu","u"),("ai","e"),("ae","e"),("ei","e"),
                 ("ey","e"),("ou","u"),("w","v"),("q","k"),("z","j"),("ck","k"),("c","k"),("d","t"),("y ","i "),
                 ("x","ks")):
        s = s.replace(a, b)
    s = re.sub(r"y$", "i", s)
    return re.sub(r"([a-z])\1+", r"\1", s)


def _clean_part(s):
    s = re.sub(r"[^a-z0-9/ ]", " ", s)
    s = re.sub(r"\b((?:[a-z] ){1,4}[a-z])\b", lambda m: m.group(0).replace(" ", ""), s)  # l l c -> llc
    toks = []
    for t in s.split():
        nd = sum(ch.isdigit() for ch in t); na = len(t) - nd
        if nd == 1 and na >= 3 and not ORD_RE.match(t):          # 5uperior -> superior
            t = t.translate(HOMO)
        toks.append(t)
    s = " ".join(toks)
    return re.sub(r"\b(\w+)( \1\b)+", r"\1", s)                  # "odyssey odyssey" -> "odyssey"


def base(s, keep_commas=False):
    s = unicodedata.normalize("NFKC", str(s))
    s = NUMSIGN_RE.sub(" no ", s)
    s = unidecode(s).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("#", " ")
    s = re.sub(r"https?://|www\.", " ", s)
    s = re.sub(r"\.(com|in|net|org|co|fr|io|biz)\b", " ", s)
    s = re.sub(r"(?<=\d)[.,](?=\d)", "", s)                      # 1,234 -> 1234
    s = JUNK_RE.sub(" ", s)
    if keep_commas:
        return " , ".join(p for p in (_clean_part(x) for x in s.split(",")) if p)
    return _clean_part(s)


def _legalize(toks):
    out = []
    for t in toks:
        out.extend(LEGAL.get(t, t).split())
    return out


# transliterated legal words (Hindi/Marathi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati)
for native, canon in [("प्राइवेट","pvt"),("लिमिटेड","ltd"),("एलएलपी","llp"),("इंक","inc"),("कंपनी","co"),
                      ("प्रा","pvt"),("लि","ltd"),("பிரைவேட்","pvt"),("லிமிடெட்","ltd"),("ప్రైవేట్","pvt"),
                      ("లిమిటెడ్","ltd"),("ಪ್ರೈವೇಟ್","pvt"),("ಲಿಮಿಟೆಡ್","ltd"),("പ്രൈവറ്റ്","pvt"),
                      ("ലിമിറ്റഡ്","ltd"),("প্রাইভেট","pvt"),("লিমিটেড","ltd"),("પ્રાઇવેટ","pvt"),("લિમિટેડ","ltd")]:
    b = base(native)
    for v in {b, re.sub(r"([a-z])\1+", r"\1", b)}:
        if v and v not in LEGAL: LEGAL[v] = canon


def norm_name(s):
    b = base(s)
    has_alias = int(bool(ALIAS_RE.search(b)))
    b = ALIAS_RE.sub(" ", b)                                     # 'x dba y' -> 'x y' (token-set sims handle subsets)
    b = re.sub(r"\b(private|pvt)\s*(limited|ltd)\b", "pvt ltd", b)
    toks = _legalize(b.split())
    toks = [re.sub(r"([a-z])\1+", r"\1", t) if not t.isdigit() else t for t in toks]
    toks = _legalize(toks)
    full = " ".join(toks)
    core = " ".join(t for t in toks if t not in NAME_STOP) or full
    legal = " ".join(sorted(set(t for t in toks if t in LEGAL_FORMS)))
    return full, core, legal, has_alias


def _find_state(comps, table, codes, alias):
    """last component first; a state is a whole component, or the tail of a later number-free component
    ('erlanger ky'). Never the tail of a street component ('12 oak ct')."""
    for i in range(len(comps) - 1, -1, -1):
        toks = comps[i]
        has_digit = any(ch.isdigit() for t in toks for ch in t)
        for n in (3, 2, 1):
            if len(toks) < n: continue
            if len(toks) > n and (has_digit or i == 0): continue
            tail = " ".join(toks[-n:])
            if tail in table: return table[tail], i, n
            if n == 1 and tail in codes: return alias.get(tail, tail), i, 1
    return "", None, 0


def norm_addr(s, country):
    c = country
    b = base(s, keep_commas=True)
    b = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", b)
    b = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z]{2,})", " ", b)   # "plot19" "17road"
    if c == "india":
        b = re.sub(r"\b(\d{3}) (\d{3})\b", r"\1\2", b)               # PIN '201 301'
    comps = [p.split() for p in b.split(",")]
    comps = [p for p in comps if p]
    # postcode: right length, never the first token of a component (that is a house number)
    zipc, zl = "", ZIP_LEN.get(c)
    if zl:
        for p in comps:
            for j in range(1, len(p)):
                if p[j].isdigit() and len(p[j]) == zl:
                    zipc = p.pop(j); break
            if zipc: break
    # state (before abbreviation expansion, so 'fl' / 'ct' are still states here)
    state = ""
    if c in STATE_TABLES:
        table, codes, alias = STATE_TABLES[c]
        state, i, n = _find_state(comps, table, codes, alias)
        if state:
            del comps[i][-n:]
            for p in comps:                                       # 'erlanger ky , ky' -> drop the repeat too
                if p and not any(ch.isdigit() for t in p for ch in t) and \
                        alias.get(p[-1], table.get(p[-1], p[-1])) == state:
                    p.pop()
    street_map = STREET_BY_COUNTRY.get(c, EN_STREET)
    toks = []
    for p in comps:
        for t in p:
            t = street_map.get(t, t)
            if c == "india": t = CITY_ALIAS.get(t, t)
            toks.append(str(int(t)) if t.isdigit() else t)      # 0200 -> 200
    full = " ".join(toks)
    street = " ".join(t for t in toks if t not in ADDR_DROP)
    nums = " ".join(re.findall(r"\d+", full))
    words = " ".join(t for t in street.split() if not t.isdigit())
    return full, street, nums, words, state, zipc


PREP_COLS = ["n_full", "n_core", "n_legal", "n_alias", "a_full", "a_street", "a_nums", "a_words",
             "a_state", "a_zip", "n_phon", "a_phon", "country_n"]


def _prep_chunk(names, addrs, countries):
    out = {k: [] for k in PREP_COLS}
    for nm, ad, co in zip(names, addrs, countries):
        cn = unidecode(str(co)).strip().lower()
        f, core, leg, al = norm_name(nm)
        af, ast, an, aw, st, zp = norm_addr(ad, cn)
        for k, v in zip(PREP_COLS, (f, core, leg, al, af, ast, an, aw, st, zp, phon(core), phon(ast), cn)):
            out[k].append(v)
    return pd.DataFrame(out)


def prepare(df, step=100_000):
    cols = [df[c].to_numpy(dtype=object) for c in ("business_name", "business_address", "country")]
    parts = Parallel(n_jobs=N_JOBS, batch_size=1)(
        delayed(_prep_chunk)(cols[0][i:i+step], cols[1][i:i+step], cols[2][i:i+step])
        for i in range(0, len(df), step))
    P = pd.concat(parts, ignore_index=True)
    out = pd.concat([df[["entity_id", "business_name", "business_address", "country"]].reset_index(drop=True), P], axis=1)
    out["n_alias"] = out["n_alias"].astype(np.int8)
    out["all_txt"] = (out["n_full"] + " " + out["a_full"]).str.strip()
    out["all_phon"] = (out["n_phon"] + " " + out["a_phon"]).str.strip()
    out["n_nospace"] = out["n_core"].str.replace(" ", "", regex=False)
    out["n_empty"] = (out["n_core"].str.len() == 0).astype(np.int8)
    out["a_empty"] = (out["a_full"].str.len() == 0).astype(np.int8)
    return out

log("normalisers defined")

# ============================================================ 2b. UNIT CHECKS (real tricky cases)
A = lambda s, c: dict(zip(["full", "street", "nums", "words", "state", "zip"], norm_addr(s, c)))
N = lambda s: dict(zip(["full", "core", "legal", "alias"], norm_name(s)))
checks = [
    # (description, got, expected)
    ("FL stays a state",            A("100 Main St, Miami, FL", "us")["state"], "fl"),
    ("CT stays a state",            A("12 Elm Rd, Hartford, CT", "us")["state"], "ct"),
    ("MT stays a state",            A("5 Oak Ave, Billings, MT", "us")["state"], "mt"),
    ("NE stays a state",            A("9 Pine Dr, Omaha, NE", "us")["state"], "ne"),
    ("Oak Ct is a street",          A("12 Oak Ct, Austin, TX", "us")["full"], "12 oak court austin"),
    ("Washington, DC -> dc",        A("2905 Rittenhouse Street, Washington, DC", "us")["state"], "dc"),
    ("District of Columbia -> dc",  A("905 Rittenhouse Saint, Washinton, District of Columbia", "us")["state"], "dc"),
    ("Kansas City, MO -> mo",       A("7915 Ames Avenue, Kansas City, MO", "us")["state"], "mo"),
    ("Kansas City, Missouri -> mo", A("7915 Ames Avenue, Kansas City, Missouri", "us")["state"], "mo"),
    ("TX first component",          A("TX, 709 Hackberry Street, Tilden", "us")["state"], "tx"),
    ("Saint == St (US)",            A("00709 Hackberry Saint, Tilden, Texas", "us")["street"],
                                    A("709 HACKBERRY ST, TILDEN, TX", "us")["street"]),
    ("zero padding",                A("00709 Hackberry Saint, Tilden, Texas", "us")["nums"], "709"),
    ("5-digit house no. is not zip", A("17560 Ellis Road, Tahlequah, OK", "us")["zip"], ""),
    ("real US zip",                 A("500 Market St, San Jose, CA 95113", "us")["zip"], "95113"),
    ("'Erlanger KY, KY' cleaned",   A("165 BARREN RIVER DR, ERLANGER KY, KY", "us")["street"], "165 barren river drive erlanger"),
    ("France: no state from 'la'",  A("12 Rue de la Paix, Paris", "france")["state"], ""),
    ("France: N° -> no",            A("N° 12 Rue Victor Hugo, Lille", "france")["nums"], "12"),
    ("France: R. -> rue",           A("63 R. DE DIEPPE, LILLE, Hauts-de-France", "france")["full"][:15], "63 rue de diepp"),
    ("France: St == Saint",         A("5 Rue X, St Nazaire", "france")["street"], A("5 Rue X, Saint-Nazaire", "france")["street"]),
    ("India PIN + state",           (A("Plot 5, Sector 18, Noida, UP 201301", "india")["zip"],
                                     A("Plot 5, Sector 18, Noida, UP 201301", "india")["state"]), ("201301", "up")),
    ("India TG",                    A("5-513/4, Cbr Estates, Hyderabad, TG", "india")["state"], "tg"),
    ("India Delhi (city kept)",     A("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "india")["state"], "dl"),
    ("India renamed city",          A("12 MG Road, Gurgaon, Haryana", "india")["street"], A("12 MG Road, Gurugram, HR", "india")["street"]),
    ("name + vs &",                 N("Elite + C0mpany")["core"], N("Elite & Co")["core"]),
    ("name leet",                   N("Cabrera 5ecure Sciences LP")["core"], N("Cabrera Secure Sciences")["core"]),
    ("name alias flag",             N("Ciracira DBA: Seven Consultants")["alias"], 1),
    ("name legal native",           N("राम मार्केटिंग प्राइवेट लिमिटेड")["legal"], "ltd pvt"),
    ("name honorific",              N("Shri Galaxy Logistics Private Limited")["core"], N("Galaxy Logistics Pvt Ltd")["core"]),
    ("name S.A.R.L",                N("Marina Ecole S.A.R.L")["legal"], "sarl"),
]
bad = 0
for desc, got, exp in checks:
    ok = got == exp
    bad += not ok
    print(f"{'OK ' if ok else 'FAIL'}  {desc:32s} got={got!r}" + ("" if ok else f"  expected={exp!r}"))
assert bad == 0, f"{bad} preprocessing checks failed"
log("all", len(checks), "preprocessing checks passed")

# ============================================================ 2c. LEARN NATIVE-SCRIPT STATE NAMES (train labels only)
NATIVE_FILE = f"{CACHE_DIR}/native_states.json"
if os.path.exists(NATIVE_FILE) and HAVE_CACHE:
    learned = json.load(open(NATIVE_FILE, encoding="utf-8"))
else:
    s1 = tr[1][tr[1]["country"].str.lower() == "india"].sample(150_000, random_state=SEED)
    s1_state = {i: norm_addr(a, "india")[4] for i, a in zip(s1["entity_id"], s1["business_address"])}
    own = {b: a for a in s1_state for b in GT.get(a, ())}
    cnt = defaultdict(Counter)
    for s in (2, 3):
        df = tr[s][tr[s]["entity_id"].isin(own)]
        for i, addr in zip(df["entity_id"], df["business_address"]):
            st = s1_state[own[i]]
            if not st: continue
            for comp in addr.split(","):
                if NATIVE_RE.search(comp):
                    k = base(comp)
                    if k: cnt[k][st] += 1
    learned = {}
    for k, c in cnt.items():
        tot = sum(c.values()); st, n = c.most_common(1)[0]
        if tot >= 15 and n / tot >= 0.9 and k not in IN_STATES:
            learned[k] = st
    json.dump(learned, open(NATIVE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
IN_STATES.update(learned)
log(f"{len(learned)} native-script state names learned:", dict(list(learned.items())[:12]))
print("check:", A("PLOT NO B-78/1, AMBERNATH EAST, THANE, महाराष्ट्र", "india")["state"],
      A("6-4, CENTRAL ROAD, KOLKATA, পশ্চিমবঙ্গ", "india")["state"])

# ============================================================ 2d. RUN PREPROCESSING ON ALL 23.4M ROWS (parallel, cached)
if HAVE_CACHE:
    PREP = {k: pd.read_parquet(p) for k, p in CACHE_FILES.items()}
    log("loaded normalised cache:", {f"{k[0]}_s{k[1]}": len(v) for k, v in PREP.items()})
else:
    PREP = {}
    for sp, d in (("train", tr), ("test", te)):
        for s in (1, 2, 3):
            t = time.time()
            df = prepare(d[s]); df["src"] = np.int8(s)
            df.to_parquet(CACHE_FILES[(sp, s)], index=False)
            PREP[(sp, s)] = df
            log(f"{sp} source{s}: {len(df):,} rows normalised in {time.time()-t:.0f}s")
    del tr, te; gc.collect()
PREP[("train", 1)][["business_name", "n_core", "n_legal", "business_address", "a_street", "a_nums", "a_state", "a_zip"]].head(5)

# ============================================================ 2e. PREPROCESSING QUALITY REPORT
# (i) coverage per split / source / country
rows = []
for (sp, s), df in PREP.items():
    for c, g in df.groupby("country_n"):
        rows.append(dict(split=sp, src=s, country=c, n=len(g), state=(g.a_state != "").mean(), zip=(g.a_zip != "").mean(),
                         nums=(g.a_nums != "").mean(), addr_empty=g.a_empty.mean(), name_empty=g.n_empty.mean(),
                         alias=g.n_alias.mean(), native_left=g.a_full.str.contains("[^a-z0-9 /]", regex=True).mean()))
cov = pd.DataFrame(rows).round(3)
print(cov.to_string(index=False))

# (ii) true pairs vs same-country random pairs on a train sample: cleaning should separate them
rng = np.random.default_rng(SEED)
QC = ["entity_id", "country_n", "n_core", "a_street", "a_nums", "a_state"]
L1 = PREP[("train", 1)][QC].sample(20_000, random_state=SEED).set_index("entity_id")
R23 = pd.concat([PREP[("train", 2)][QC], PREP[("train", 3)][QC]]).set_index("entity_id")
pairs = [(a, b) for a in L1.index for b in GT.get(a, ())]
by_c = {c: g.index.values for c, g in R23.groupby("country_n")}
neg = [(a, by_c[L1.at[a, "country_n"]][rng.integers(len(by_c[L1.at[a, "country_n"]]))]) for a, _ in pairs[:20_000]]
def prof(pp):
    la, rb = L1.loc[[a for a, _ in pp]], R23.loc[[b for _, b in pp]]
    ns = cpdist(la.n_core.tolist(), rb.n_core.tolist(), scorer=fuzz.token_set_ratio, workers=-1).mean()
    ads = cpdist(la.a_street.tolist(), rb.a_street.tolist(), scorer=fuzz.token_set_ratio, workers=-1).mean()
    both = [(x, y) for x, y in zip(la.a_nums, rb.a_nums) if x and y]
    num = np.mean([bool(set(x.split()) & set(y.split())) for x, y in both])
    st = [(x, y) for x, y in zip(la.a_state, rb.a_state) if x and y]
    return ns, ads, num, np.mean([x == y for x, y in st]), len(st) / len(pp)
print(f"\n{'':12s} {'name sim':>9s} {'addr sim':>9s} {'nums overlap':>13s} {'state equal':>12s} {'state known':>12s}")
for lbl, pp in (("true pairs", pairs), ("random", neg)):
    print(f"{lbl:12s} " + " ".join(f"{v:>{w}.3f}" for v, w in zip(prof(pp), (9, 9, 13, 12, 12))))

# (iii) remaining state disagreements on true pairs (should be genuine vendor noise, not our bugs)
mis = Counter((L1.at[a, "country_n"], L1.at[a, "a_state"], R23.at[b, "a_state"]) for a, b in pairs
              if L1.at[a, "a_state"] and R23.at[b, "a_state"] and L1.at[a, "a_state"] != R23.at[b, "a_state"])
print("\nstate disagreements on true pairs:", sum(mis.values()), "->", mis.most_common(8))

# (iv) France (test only, no labels): frequent S2/S3 address tokens that are rare in S1 = unhandled variants?
fr = {s: PREP[("test", s)].query("country_n == 'france'") for s in (1, 2, 3)}
v1 = Counter(t for x in fr[1].a_full for t in x.split())
v23 = Counter(t for s in (2, 3) for x in fr[s].a_full.sample(200_000, random_state=SEED) for t in x.split())
odd = [(t, c) for t, c in v23.most_common(4000) if v1[t] * 8 < c * 0.02 and not t.isdigit() and len(t) < 12]
print("\nFrance: frequent S2/S3 address tokens rare in S1:", odd[:25])

# ============================================================ 3. SAMPLE FOR THIS RUN
_R_FULL = {}
def full_R(split):
    """all S2+S3 records of a split (built once, reused)"""
    if split not in _R_FULL:
        _R_FULL[split] = pd.concat([PREP[(split, 2)], PREP[(split, 3)]], ignore_index=True)
    return _R_FULL[split]

def make_L(split, n_s1, seed=SEED):
    L = PREP[(split, 1)]
    if n_s1 and n_s1 < len(L): L = L.sample(n=n_s1, random_state=seed)
    return L.reset_index(drop=True)

# DEV = sample of S1 against ALL of S2/S3 (realistic decoys); FULL = train sample / all test S1
trL, trR = make_L("train", DEV_TRAIN_S1 if MODE == "DEV" else FULL_TRAIN_S1), full_R("train")
teL, teR = make_L("test", DEV_TEST_S1 if MODE == "DEV" else None), full_R("test")
# the S2/S3 frames now live (once) inside _R_FULL; drop the per-source copies to save ~12 GB
for sp_ in ("train", "test"): PREP[(sp_, 2)] = PREP[(sp_, 3)] = None
gc.collect()
log(f"train L={len(trL):,} R={len(trR):,} | test L={len(teL):,} R={len(teR):,}")

# ============================================================ 4. EMBEDDINGS (optional)
EMB = {}
if USE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer(EMB_MODEL, device=dev)
        enc = lambda df: st_model.encode((df["business_name"] + " | " + df["business_address"]).tolist(),
                                         batch_size=512, convert_to_numpy=True, normalize_embeddings=True,
                                         show_progress_bar=False).astype(np.float32)
        for tag, L, R in (("tr", trL, trR), ("te", teL, teR)):
            EMB[tag + "L"], EMB[tag + "R"] = enc(L), enc(R)
        log("embeddings done on", dev); del st_model; gc.collect()
    except Exception as e:
        log("!! embeddings disabled:", repr(e)); USE_EMBEDDINGS = False
else:
    log("embeddings off")

# ============================================================ 5a. BLOCKING — definitions
# Speed: (1) compare only inside country + state groups (states agree in 99.3% of true pairs;
# records without a state are compared with their whole country; TG/AP merged), (2) 3-4 char n-grams
# with common ones pruned (max_df), (3) cheap exact-key joins for what n-grams miss.
# 5b measured: v_all_c (3-5 grams) cost 58% of the time for 0.03% unique recall -> removed.
# Candidates are ranked by reciprocal-rank fusion over the views (better than summing scores).
import scipy.sparse as sp
VIEWS = {  # name: (column, analyzer, ngram range, max_df)
    "v_name_c": ("n_core",   "char_wb", (3, 4), 0.02),
    "v_addr_c": ("a_street", "char_wb", (3, 4), 0.02),
    "v_all_w":  ("all_txt",  "word",    (1, 2), 0.01),
    "v_phon_c": ("all_phon", "char_wb", (3, 4), 0.02),
}
KEY_VIEWS = ["v_key_addr", "v_key_name"]
ALL_V = list(VIEWS) + ["v_emb"] + KEY_VIEWS
STATE_GROUP = {"tg": "tg_ap", "ap": "tg_ap"}
KEY_MAX_BUCKET = 50        # exact keys shared by more R records than this are too generic -> skipped
FIT_SAMPLE = 2_000_000     # vocabulary / idf fitted on a sample of R (+ all of L)

def _to_np(s): return s.to_numpy(dtype=object)

def _transform(vec, texts, step=400_000):
    parts = Parallel(n_jobs=min(N_JOBS, 8))(delayed(vec.transform)(texts[i:i+step])
                                           for i in range(0, len(texts), step))
    return sp.vstack(parts).tocsr()

def fit_views(L, R, views=VIEWS):
    mats = {}
    rs = R.sample(min(len(R), FIT_SAMPLE), random_state=SEED)
    for v, (col, an, ng, mx) in views.items():
        t = time.time()
        vec = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=2, max_df=mx, sublinear_tf=True, dtype=np.float32)
        vec.fit(np.concatenate([_to_np(L[col]), _to_np(rs[col])]))
        mats[v] = (_transform(vec, _to_np(L[col])), _transform(vec, _to_np(R[col])))
        log(f"   view {v:9s}: vocab={len(vec.vocabulary_):,}  ({time.time()-t:.0f}s)")
    return mats

def _groups(df):
    st = df["a_state"].to_numpy(dtype=object)
    return df["country_n"].to_numpy(dtype=object), np.array([STATE_GROUP.get(s, s) for s in st], dtype=object)

def partitions(L, R):
    """yield (L row indices, boolean mask over R) for each country / state group"""
    Lc, Ls = _groups(L); Rc, Rs = _groups(R)
    for c in np.unique(Lc):
        lc = Lc == c; rc = Rc == c
        if c not in STATE_TABLES:                    # France / unseen country: whole country
            yield np.flatnonzero(lc), rc; continue
        r_nostate = rc & (Rs == "")
        for s in np.unique(Ls[lc]):
            lidx = np.flatnonzero(lc & (Ls == s))
            yield lidx, (rc if s == "" else (rc & (Rs == s)) | r_nostate)

def _key_pairs(lk, rk, cap=KEY_MAX_BUCKET):
    ldf = pd.DataFrame({"k": lk, "li": np.arange(len(lk))}); ldf = ldf[ldf.k != ""]
    rdf = pd.DataFrame({"k": rk, "ri": np.arange(len(rk))}); rdf = rdf[rdf.k != ""]
    sz = rdf["k"].map(rdf["k"].value_counts()); rdf = rdf[sz.values <= cap]
    m = ldf.merge(rdf, on="k")
    return m["li"].values, m["ri"].values

def _addr_key(df):   # country | first house number | first street word
    num0 = df["a_nums"].str.split(n=1).str[0].fillna("")
    w0 = df["a_words"].str.split(n=1).str[0].fillna("")
    k = (df["country_n"] + "|" + num0 + "|" + w0).to_numpy(dtype=object)
    k[(num0 == "").values | (w0 == "").values] = ""
    return k

def _name_key(df, dfreq):   # country | rarest name token (len >= 3)
    out = []
    for c, core in zip(df["country_n"].values, df["n_core"].values):
        toks = [t for t in core.split() if len(t) >= 3]
        out.append(c + "|" + min(toks, key=lambda t: dfreq.get(t, 0)) if toks else "")
    return np.array(out, dtype=object)

def key_block(L, R):
    dfreq = pd.concat([L["n_core"], R["n_core"]]).str.split().explode().value_counts().to_dict()
    return {"v_key_addr": _key_pairs(_addr_key(L), _addr_key(R)),
            "v_key_name": _key_pairs(_name_key(L, dfreq), _name_key(R, dfreq))}

def block(L, R, mats, tag, views=VIEWS, use_keys=True, k=K_TFIDF, timing=None):
    nR, src_r = len(R), R["src"].values
    vid = {v: i for i, v in enumerate(ALL_V)}
    LI, RI, SC, VW = [], [], [], []
    for lidx, rmask in partitions(L, R):
        for src in (2, 3):
            ridx = np.flatnonzero(rmask & (src_r == src))
            if len(ridx) == 0: continue
            for v in views:
                t = time.time()
                A_, B_ = mats[v]
                C = sp_matmul_topn(A_[lidx], B_[ridx].T.tocsr(), top_n=k, threshold=1e-6, n_threads=N_JOBS).tocoo()
                LI.append(lidx[C.row]); RI.append(ridx[C.col]); SC.append(C.data.astype(np.float32))
                VW.append(np.full(len(C.data), vid[v], np.int8))
                if timing is not None: timing[v] += time.time() - t
            if USE_EMBEDDINGS:
                ii, ss = topk_dense(EMB[tag + "L"][lidx], EMB[tag + "R"][ridx], K_EMB)
                LI.append(np.repeat(lidx, ii.shape[1])); RI.append(ridx[ii.ravel()])
                SC.append(ss.ravel().astype(np.float32)); VW.append(np.full(ii.size, vid["v_emb"], np.int8))
    if use_keys:
        t = time.time()
        for v, (li_, ri_) in key_block(L, R).items():
            LI.append(li_); RI.append(ri_); SC.append(np.ones(len(li_), np.float32))
            VW.append(np.full(len(li_), vid[v], np.int8))
        if timing is not None: timing["keys"] += time.time() - t
    li, ri, sc, vw = map(np.concatenate, (LI, RI, SC, VW))
    keep = sc > 0; li, ri, sc, vw = li[keep], ri[keep], sc[keep], vw[keep]
    key = li.astype(np.int64) * nR + ri
    uk, inv = np.unique(key, return_inverse=True)
    W = np.full((len(uk), len(ALL_V)), np.nan, np.float32); W[inv, vw] = sc
    P = pd.DataFrame(W, columns=ALL_V)
    P.insert(0, "li", (uk // nR).astype(np.int32)); P.insert(1, "ri", (uk % nR).astype(np.int32))
    P["n_views"] = np.isfinite(W).sum(1).astype(np.int8)
    P["bsum"] = np.nan_to_num(W).sum(1)
    src = src_r[P["ri"].values]
    rrf = np.zeros(len(P))
    for v in views:                                     # reciprocal-rank fusion within (S1, source)
        s = P[v].fillna(-1).values
        r = pd.Series(s).groupby([P["li"].values, src]).rank(ascending=False, method="first").values
        rrf += np.where(s > 0, 1.0 / (5 + r), 0)
    rrf += 0.2 * P["v_key_addr"].notna().values + 0.1 * P["v_key_name"].notna().values
    P["brrf"] = rrf.astype(np.float32)
    P = P.sort_values(["li", "brrf"], ascending=[True, False]).reset_index(drop=True)
    P["brank"] = P.groupby("li").cumcount().values.astype(np.int16)
    return P

def cap_pairs(P, cap):
    return P[P["brank"].values < cap].reset_index(drop=True)

def topk_dense(A, B, k, chunk=4096):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.from_numpy(B).to(dev); ri, rs = [], []
    for st in range(0, len(A), chunk):
        S = torch.from_numpy(A[st:st+chunk]).to(dev) @ Bt.T
        s, i = torch.topk(S, min(k, B.shape[0]), dim=1)
        ri.append(i.cpu().numpy()); rs.append(s.cpu().numpy())
    return np.vstack(ri), np.vstack(rs)

def label_pairs(P, L, R):
    rpos = pd.Series(np.arange(len(R)), index=R["entity_id"].values); nR = len(R)
    gl, gr = [], []
    for l, s in enumerate(L["entity_id"].values):
        for x in GT.get(s, ()):
            gl.append(l); gr.append(x)
    gr_i = rpos.reindex(gr).values
    ok = ~np.isnan(gr_i)
    gk = np.array(gl)[ok].astype(np.int64) * nR + gr_i[ok].astype(np.int64)
    key = P["li"].values.astype(np.int64) * nR + P["ri"].values
    return np.isin(key, gk).astype(np.int8), len(gl)
log("blocking functions defined")

if RUN_BLOCK_EXPERIMENT:
    # ============================================================ 5b. BLOCKING EXPERIMENT (20k train S1 vs ALL 10.3M train S2/S3)
    EXP_N = 20_000
    expL, expR = make_L("train", EXP_N, seed=SEED + 1), full_R("train")
    t0 = time.time(); expM = fit_views(expL, expR); t_fit = time.time() - t0
    timing = Counter(); t0 = time.time()
    expP = block(expL, expR, expM, "exp", timing=timing); t_blk = time.time() - t0
    expP["y"], exp_tot = label_pairs(expP, expL, expR)
    log(f"fit {t_fit:.0f}s | block {t_blk:.0f}s  per view: " + ", ".join(f"{k}={v:.0f}s" for k, v in timing.items()))
    log(f"pairs={len(expP):,} ({len(expP)/EXP_N:.1f}/S1) | recall={expP.y.sum()/exp_tot:.4f}  ({expP.y.sum():,}/{exp_tot:,})")

    vcols = [v for v in ALL_V if expP[v].notna().any()]
    found = expP[vcols].notna().values; yy = expP.y.values.astype(bool)
    print(f"\n{'view':12s} {'recall':>7s} {'unique':>7s}   (unique = true pairs ONLY this view finds)")
    for j, v in enumerate(vcols):
        only = found[:, j] & (found.sum(1) == 1)
        print(f"{v:12s} {(found[:, j] & yy).sum()/exp_tot:7.4f} {(only & yy).sum()/exp_tot:7.4f}")

    print("\nrecall by cap (candidates kept per S1, ranked by blocking score):")
    for cap in (5, 10, 15, 20, 25, 30, 40, 60, 100):
        sel = expP.brank.values < cap
        print(f"  cap {cap:3d}: recall={expP.y.values[sel].sum()/exp_tot:.4f}  pairs/S1={sel.sum()/EXP_N:5.1f}")

    cn = expL["country_n"].values[expP.li.values]
    tot_c = Counter(expL["country_n"].values[l] for l, s in enumerate(expL.entity_id) for _ in GT.get(s, ()))
    print("\nrecall by country:", {c: round(expP.y.values[cn == c].sum() / n, 4) for c, n in tot_c.items()})

    per_row = t_blk / EXP_N
    log(f"extrapolated blocking time: train {FULL_TRAIN_S1:,} S1 ~ {per_row*FULL_TRAIN_S1/60:.0f} min | "
        f"test {len(PREP[('test', 1)]):,} S1 ~ {per_row*len(PREP[('test', 1)])/60:.0f} min")

    # which true pairs does blocking miss?  (look at them to improve)
    found_keys = set(zip(expP.li.values[yy], expP.ri.values[yy]))
    rpos = dict(zip(expR.entity_id.values, range(len(expR))))
    missed = [(l, rpos[x]) for l, s in enumerate(expL.entity_id) for x in GT.get(s, ()) if (l, rpos[x]) not in found_keys]
    print(f"\nmissed {len(missed):,} true pairs, examples:")
    for l, r in missed[:12]:
        a, b = expL.iloc[l], expR.iloc[r]
        print(f"  S1 {a.business_name[:35]:35s} | {a.business_address[:55]:55s} [{a.a_state}]")
        print(f"   ->{b.business_name[:35]:35s} | {b.business_address[:55]:55s} [{b.a_state}]")

# ============================================================ 5c. BLOCKING — train
for v in ("expM", "expP"): globals().pop(v, None)
gc.collect()
log("blocking train ...")
trM = fit_views(trL, trR)
trP = cap_pairs(block(trL, trR, trM, "tr"), MAX_CAND_PER_S1)
trP["y"], tot_true = label_pairs(trP, trL, trR)
log(f"train pairs={len(trP):,} ({len(trP)/len(trL):.1f}/S1) | blocking recall={trP.y.sum()/max(tot_true,1):.4f}")

# ============================================================ 6. FEATURES
def rowdot(A, B, li, ri, bs=500_000):
    out = np.zeros(len(li), np.float32)
    for st in range(0, len(li), bs):
        out[st:st+bs] = np.asarray(A[li[st:st+bs]].multiply(B[ri[st:st+bs]]).sum(1)).ravel()
    return out

def idf_table(L, R, col):
    df = Counter()
    for x in pd.concat([L[col], R[col]]).values:
        df.update(set(x.split()))
    Nd = len(L) + len(R)
    return {t: math.log((Nd + 1) / (c + 1)) + 1 for t, c in df.items()}

def second_max(x, key):
    """vectorised 2nd-largest value of x within each key group (-1 if group size 1)."""
    x = np.asarray(x, np.float64); key = np.asarray(key)
    o = np.lexsort((-x, key)); ks = key[o]
    first = np.r_[True, ks[1:] != ks[:-1]]
    pos = np.arange(len(o)) - np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
    sec_val = pd.Series(np.where(pos == 1, x[o], -np.inf)).groupby(ks).transform("max").values
    out = np.empty(len(x)); out[o] = np.where(np.isinf(sec_val), -1, sec_val)
    return out

def add_competition(F, P, cols, pref=""):
    li = pd.Series(P["li"].values, index=F.index); ri = pd.Series(P["ri"].values, index=F.index)
    for c in cols:
        x = F[c]
        key = li.astype(np.int64) * 10 + F["src"].astype(np.int64)
        F[f"{pref}{c}_rank_l"] = x.groupby(key).rank(ascending=False, method="min")
        F[f"{pref}{c}_gap_l"] = x - x.groupby(li).transform("max")
        mx = x.groupby(ri).transform("max")
        sec = second_max(x.values, ri.values)
        F[f"{pref}{c}_gap_r"] = np.where(x >= mx, x - sec, x - mx)
        F[f"{pref}{c}_rank_r"] = x.groupby(ri).rank(ascending=False, method="min")

def features(L, R, P, mats, tag, idf=None):
    li, ri = P["li"].values, P["ri"].values
    F = pd.DataFrame(index=P.index)
    for c in ALL_V + ["n_views", "bsum", "brrf"]:
        F["blk_" + c] = P[c].values
    for v in VIEWS:
        F["cos_" + v] = rowdot(mats[v][0], mats[v][1], li, ri)
    if USE_EMBEDDINGS:
        F["cos_emb"] = np.einsum("ij,ij->i", EMB[tag + "L"][li], EMB[tag + "R"][ri])
    W = dict(workers=-1, dtype=np.float32)
    def pair(col_l, col_r=None):
        return L[col_l].values[li].tolist(), R[col_r or col_l].values[ri].tolist()
    a, b = pair("n_core")
    F["n_ratio"] = cpdist(a, b, scorer=fuzz.ratio, **W)
    F["n_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, **W)
    F["n_tsort"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, **W)
    F["n_partial"] = cpdist(a, b, scorer=fuzz.partial_ratio, **W)
    F["n_jw"] = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, **W)
    F["n_lev"] = cpdist(a, b, scorer=Levenshtein.distance, **W)
    a2, b2 = pair("n_full"); F["n_full_tset"] = cpdist(a2, b2, scorer=fuzz.token_set_ratio, **W)
    a2, b2 = pair("n_phon"); F["n_phon_ratio"] = cpdist(a2, b2, scorer=fuzz.token_sort_ratio, **W)
    a2, b2 = pair("n_nospace"); F["n_nospace_ratio"] = cpdist(a2, b2, scorer=fuzz.ratio, **W)
    F["n_nospace_partial"] = cpdist(a2, b2, scorer=fuzz.partial_ratio, **W)
    a, b = pair("a_street")
    F["a_ratio"] = cpdist(a, b, scorer=fuzz.ratio, **W)
    F["a_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, **W)
    F["a_tsort"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, **W)
    F["a_partial"] = cpdist(a, b, scorer=fuzz.partial_ratio, **W)
    a2, b2 = pair("a_phon"); F["a_phon_tsort"] = cpdist(a2, b2, scorer=fuzz.token_sort_ratio, **W)
    a2, b2 = pair("all_txt"); F["all_tset"] = cpdist(a2, b2, scorer=fuzz.token_set_ratio, **W)
    F["all_tsort"] = cpdist(a2, b2, scorer=fuzz.token_sort_ratio, **W)
    a2, b2 = pair("n_core", "a_street"); F["x_name_addr"] = cpdist(a2, b2, scorer=fuzz.token_set_ratio, **W)
    a2, b2 = pair("a_street", "n_core"); F["x_addr_name"] = cpdist(a2, b2, scorer=fuzz.token_set_ratio, **W)
    idf_n, idf_a = idf if idf is not None else (idf_table(L, R, "n_core"), idf_table(L, R, "a_words"))
    Ln, Rn = L["n_core"].values, R["n_core"].values
    La, Ra = L["a_words"].values, R["a_words"].values
    Lnum, Rnum = L["a_nums"].values, R["a_nums"].values
    Lzip, Rzip = L["a_zip"].values, R["a_zip"].values
    Lleg, Rleg = L["n_legal"].values, R["n_legal"].values
    Lst, Rst = L["a_state"].values, R["a_state"].values
    Lc, Rc = L["country_n"].values, R["country_n"].values
    out = np.zeros((len(li), 18), np.float32)
    for k, (l, r) in enumerate(zip(li, ri)):
        ta, tb = Ln[l].split(), Rn[r].split()
        sa, sb = set(ta), set(tb); inter = sa & sb; uni = sa | sb
        wi = sum(idf_n.get(t, 1) for t in inter); wu = sum(idf_n.get(t, 1) for t in uni) or 1
        out[k, 0] = wi / wu
        out[k, 1] = max((idf_n.get(t, 1) for t in inter), default=0)
        out[k, 2] = max((idf_n.get(t, 1) for t in (uni - inter)), default=0)
        out[k, 3] = bool(ta) and bool(tb) and ta[0] == tb[0]
        wa, wb = set(La[l].split()), set(Ra[r].split()); ia = wa & wb; ua = wa | wb
        out[k, 4] = sum(idf_a.get(t, 1) for t in ia) / (sum(idf_a.get(t, 1) for t in ua) or 1)
        out[k, 5] = max((idf_a.get(t, 1) for t in ia), default=0)
        na, nb = Lnum[l].split(), Rnum[r].split(); sna, snb = set(na), set(nb)
        out[k, 6] = len(sna & snb)
        out[k, 7] = len(sna & snb) / len(sna | snb) if (sna or snb) else -1
        out[k, 8] = -1 if not na or not nb else float(na[0] == nb[0])
        out[k, 9] = -1 if not na or not nb else float(na[0] in snb or nb[0] in sna)
        out[k, 10] = -1 if not Lzip[l] or not Rzip[r] else float(Lzip[l] == Rzip[r])
        out[k, 11] = len(sna ^ snb)
        out[k, 12] = -1 if not Lst[l] or not Rst[r] else float(Lst[l] == Rst[r])
        out[k, 13] = -1 if not Lleg[l] or not Rleg[r] else float(Lleg[l] == Rleg[r])
        out[k, 14] = float(Lc[l] == Rc[r])
        out[k, 15] = len(sa); out[k, 16] = len(sb)
        out[k, 17] = abs(len(wa) - len(wb))
    names = ["n_idf_jac","n_idf_maxshared","n_idf_maxunshared","n_first_eq","a_idf_jac","a_idf_maxshared",
             "num_shared","num_jac","num_first_eq","num_first_in","zip_eq","num_symdiff","state_eq","legal_eq",
             "country_eq","n_ntok_l","n_ntok_r","a_nwords_diff"]
    for j, nm in enumerate(names): F[nm] = out[:, j]
    F["src"] = R["src"].values[ri].astype(np.int8)
    F["l_a_empty"] = L["a_empty"].values[li]; F["r_a_empty"] = R["a_empty"].values[ri]
    F["r_n_empty"] = R["n_empty"].values[ri]; F["r_alias"] = R["n_alias"].values[ri]
    F["len_n_r"] = R["n_core"].str.len().values[ri]; F["len_n_l"] = L["n_core"].str.len().values[li]
    cos_cols = ["cos_v_name_c", "cos_v_addr_c", "cos_v_all_w", "cos_v_phon_c"] + (["cos_emb"] if USE_EMBEDDINGS else [])
    F["combo"] = F[cos_cols].mean(1)
    add_competition(F, P, ["combo", "cos_v_name_c", "cos_v_addr_c", "n_tset", "a_tset"])
    F["n_cand"] = P.groupby("li")["ri"].transform("size").values
    # empty text -> similarity is unknown (NaN), not a perfect match (rapidfuzz gives 100 for "" vs "")
    le_n = L["n_core"].str.len().values[li] == 0; re_n = R["n_core"].str.len().values[ri] == 0
    le_a = L["a_street"].str.len().values[li] == 0; re_a = R["a_street"].str.len().values[ri] == 0
    m_n, m_a = le_n | re_n, le_a | re_a
    for c in ["n_ratio","n_tset","n_tsort","n_partial","n_jw","n_lev","n_full_tset","n_phon_ratio",
              "n_nospace_ratio","n_nospace_partial","x_name_addr","n_idf_jac"]:
        F.loc[m_n, c] = np.nan
    for c in ["a_ratio","a_tset","a_tsort","a_partial","a_phon_tsort","x_addr_name","a_idf_jac","cos_v_addr_c"]:
        F.loc[m_a, c] = np.nan
    F["both_a_empty"] = (le_a & re_a).astype(np.int8)
    F["both_n_empty"] = (le_n & re_n).astype(np.int8)
    return F

log("features train ..."); trF = features(trL, trR, trP, trM, "tr")
FEATS = list(trF.columns)
del trM; gc.collect()
log(f"n_features={len(FEATS)}")

# ============================================================ 7. METRIC
def f05_macro(pred, truth, ids):
    """exact challenge metric (dict s1 -> set), incl. singletons."""
    sc = []
    for s in ids:
        p, t = pred.get(s, set()), truth.get(s, set())
        if not t and not p: sc.append(1.0); continue
        if not t or not p: sc.append(0.0); continue
        tp = len(p & t)
        if tp == 0: sc.append(0.0); continue
        pr, rc = tp / len(p), tp / len(t)
        sc.append(1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(sc))

TRUE_N = np.array([len(GT.get(s, ())) for s in trL["entity_id"].values], np.float64)

def f05_mask(P, keep, y, true_n=TRUE_N, subset=None):
    """same metric, vectorised: F = 1.25*TP / (0.25*|truth| + |pred|); empty & singleton -> 1."""
    li = P["li"].values[keep]
    k = np.bincount(li, minlength=len(true_n)).astype(np.float64)
    tp = np.bincount(li, weights=y[keep], minlength=len(true_n))
    F = np.where(tp > 0, 1.25 * tp / (0.25 * true_n + k), 0.0)
    F[(true_n == 0) & (k == 0)] = 1.0
    return float(F.mean() if subset is None else F[subset].mean())

log(f"(reference) predict-all-empty F0.5 = {f05_mask(trP, np.zeros(len(trP), bool), trP.y.values):.5f}")

# ============================================================ 8. STAGE-1 GBM on the GPU (XGBoost, device="cuda")
import xgboost as xgb
GBM_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                  learning_rate=0.08, grow_policy="lossguide", max_leaves=127, max_depth=0,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=5,
                  max_bin=256, seed=SEED)
PRED_CHUNK = 2_000_000   # rows per GPU prediction batch (8 GB card)

def train_cv(X, y, groups, params=GBM_PARAMS, rounds=4000):
    cols = list(X.columns); Xn = X.to_numpy(np.float32)
    oof = np.zeros(len(X)); models = []; imps = []
    for f, (a, b) in enumerate(GroupKFold(n_splits=N_FOLDS).split(Xn, y, groups)):
        t = time.time()
        dtr = xgb.QuantileDMatrix(Xn[a], y[a], feature_names=cols)
        dva = xgb.QuantileDMatrix(Xn[b], y[b], ref=dtr, feature_names=cols)
        m = xgb.train(params, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=False)
        oof[b] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        models.append(m); imps.append(pd.Series(m.get_score(importance_type="gain")))
        del dtr, dva; gc.collect()
        log(f"   fold {f}: best_it={m.best_iteration}  ({time.time()-t:.0f}s on {params['device']})")
    return oof, models, pd.concat(imps, axis=1).fillna(0).mean(1).sort_values(ascending=False)

def predict(models, X):
    Xn = X.to_numpy(np.float32); out = np.zeros(len(Xn))
    for st in range(0, len(Xn), PRED_CHUNK):
        part = Xn[st:st + PRED_CHUNK]
        out[st:st + PRED_CHUNK] = np.mean([m.inplace_predict(part, iteration_range=(0, m.best_iteration + 1))
                                           for m in models], axis=0)
    return out

y = trP["y"].values; groups = trP["li"].values
log("stage-1 training (GPU) ...")
oof1, models1, imp1 = train_cv(trF[FEATS], y, groups)
log(f"stage-1 OOF AUC={roc_auc_score(y, oof1):.5f}  AP={average_precision_score(y, oof1):.5f}")
imp1.head(20)

# ============================================================ 9. STAGE-2 (cluster features)
def build_all_vec(L, R):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=2, max_df=0.05,
                          dtype=np.float32)
    vec.fit(np.concatenate([_to_np(L["all_phon"]), _to_np(R["all_phon"].sample(min(len(R), FIT_SAMPLE), random_state=SEED))]))
    return _transform(vec, _to_np(R["all_phon"]))

def stage2_feats(F, P, p1, RV):
    G = F.copy(); G["p1"] = p1
    li = P["li"].values; ri = P["ri"].values
    s = pd.Series(p1, index=F.index); gl = s.groupby(li)
    G["p1_max_l"] = gl.transform("max"); G["p1_sum_l"] = gl.transform("sum")
    G["p1_gap_l"] = s - G["p1_max_l"]; G["p1_rank_l"] = gl.rank(ascending=False, method="min")
    G["p1_n05_l"] = (s > 0.5).groupby(li).transform("sum")
    gr = s.groupby(ri); mx = gr.transform("max")
    sec = np.maximum(second_max(p1, ri), 0)
    G["p1_gap_r"] = np.where(s >= mx, s - sec, s - mx); G["p1_nS1_r"] = gr.transform("size")
    sup_max = np.zeros(len(P), np.float32); sup_w = np.zeros(len(P), np.float32)
    order = np.argsort(li, kind="stable"); li_s = li[order]
    bounds = np.flatnonzero(np.r_[True, li_s[1:] != li_s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 2: continue
        V = RV[ri[idx]]; S = (V @ V.T).toarray(); np.fill_diagonal(S, 0)
        pp = p1[idx]; W_ = S * pp[None, :]
        sup_max[idx] = W_.max(1); sup_w[idx] = W_.sum(1) / (pp.sum() - pp + 1e-6)
    G["sup_max"] = sup_max; G["sup_w"] = sup_w
    return G

log("stage-2 features ...")
trRV = build_all_vec(trL, trR)
trG = stage2_feats(trF, trP, oof1, trRV)
FEATS2 = list(trG.columns)
del trRV; gc.collect()
log("stage-2 training ...")
oof2, models2, imp2 = train_cv(trG[FEATS2], y, groups)
log(f"stage-2 OOF AUC={roc_auc_score(y, oof2):.5f}  AP={average_precision_score(y, oof2):.5f}")
del trG; gc.collect()
imp2.head(15)

# ============================================================ 10. DECISION LAYER (tuned on macro-F0.5)
def one_to_one(P, p):
    """each S2/S3 record may belong to at most one S1 -> zero out non-best owners."""
    best = pd.Series(p).groupby(P["ri"].values).transform("max").values
    return np.where(p >= best - 1e-12, p, 0.0)

def mask_thresh(P, p, t_gate, t_inc):
    """empty-list gate on max prob, then include candidates >= t_inc (always the top one if gated in)."""
    mx = pd.Series(p).groupby(P["li"].values).transform("max").values
    return (mx >= t_gate) & ((p >= t_inc) | (p >= mx))

def mask_expected(P, p, n_samp=400, rng=None):
    """Monte-Carlo expected-F0.5-optimal top-k per S1 (k=0 allowed => singleton handling)."""
    rng = rng or np.random.default_rng(0)
    df = pd.DataFrame({"li": P["li"].values, "p": p}).sort_values(["li", "p"], ascending=[True, False])
    keep = np.zeros(len(P), bool)
    lis = df["li"].values; bounds = np.flatnonzero(np.r_[True, lis[1:] != lis[:-1], True])
    idx_all, p_all = df.index.values, df["p"].values
    for a, b in zip(bounds[:-1], bounds[1:]):
        pp = p_all[a:b][:30]
        if pp[0] < 0.02: continue
        S = rng.random((n_samp, len(pp))) < pp[None, :]
        T = S.sum(1); tpk = np.cumsum(S, 1); k = np.arange(1, len(pp) + 1)[None, :]
        e = (1.25 * tpk / (0.25 * T[:, None] + k)).mean(0)
        kb = int(np.argmax(e))
        if e[kb] > np.mean(T == 0): keep[idx_all[a:a + kb + 1]] = True
    return keep

results = []
for use121 in (False, True):
    p = one_to_one(trP, oof2) if use121 else oof2
    best = (0, None)
    for tg in np.arange(0.20, 0.91, 0.05):
        for ti in np.arange(0.10, tg + 1e-9, 0.05):
            sc = f05_mask(trP, mask_thresh(trP, p, tg, ti), y)
            if sc > best[0]: best = (sc, (float(round(tg, 2)), float(round(ti, 2))))
    results.append(("thresh", use121, best[1], best[0]))
    log(f"thresh   one2one={use121}: best F0.5={best[0]:.5f} @ gate,inc={best[1]}")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    sc = f05_mask(trP, mask_expected(trP, iso.predict(p)), y)
    results.append(("expected", use121, None, sc))
    log(f"expected one2one={use121}: F0.5={sc:.5f}")
best_cfg = max(results, key=lambda x: x[3])
log("BEST decision rule:", best_cfg)

def apply_rule(P, p_raw, cfg, p_ref, y_ref):
    kind, use121, prm, _ = cfg
    p = one_to_one(P, p_raw) if use121 else p_raw
    if kind == "thresh": return mask_thresh(P, p, *prm)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ref, y_ref)
    return mask_expected(P, iso.predict(p))

p_ref = one_to_one(trP, oof2) if best_cfg[1] else oof2
keep_tr = apply_rule(trP, oof2, best_cfg, p_ref, y)
for c in np.unique(trL["country_n"].values):
    sub = trL["country_n"].values == c
    log(f"   OOF F0.5 country={c!r:10s} n={sub.sum():7,d}: {f05_mask(trP, keep_tr, y, subset=sub):.5f}")
log(f"   OOF F0.5 singletons={f05_mask(trP, keep_tr, y, subset=TRUE_N == 0):.5f} | "
    f"non-singletons={f05_mask(trP, keep_tr, y, subset=TRUE_N > 0):.5f}")
# cross-check the vectorised metric against the exact dict-based one
ids_l, ids_r = trL["entity_id"].values, trR["entity_id"].values
pred_tr = defaultdict(set)
for l, r in zip(trP["li"].values[keep_tr], trP["ri"].values[keep_tr]): pred_tr[ids_l[l]].add(ids_r[r])
log(f"   exact metric check: {f05_macro(pred_tr, GT, list(ids_l)):.5f}")

# ============================================================ 10b. TEST INFERENCE (one country at a time -> bounded memory)
# No true match crosses countries (0 of 7.6M in train), and each S2/S3 record belongs to one country,
# so blocking, reverse-competition features, stage-2 and one-to-one are all exact per country.
for v in ("trF", "trM"): globals().pop(v, None)
gc.collect()
log("fitting test views ..."); teM = fit_views(teL, teR)
teRV = build_all_vec(teL, teR)
IDF_TE = (idf_table(teL, teR, "n_core"), idf_table(teL, teR, "a_words"))
ids_l, ids_r = teL["entity_id"].values, teR["entity_id"].values
pred_te, cand_te = defaultdict(set), defaultdict(set)
for c in np.unique(teL["country_n"].values):
    t = time.time()
    lmask = teL["country_n"].values == c
    Lc = teL[lmask].reset_index(drop=True); lpos = np.flatnonzero(lmask)
    Pc = cap_pairs(block(Lc, teR, {v: (teM[v][0][lpos], teM[v][1]) for v in VIEWS}, "te"), MAX_CAND_PER_S1)
    Fc = features(Lc, teR, Pc, {v: (teM[v][0][lpos], teM[v][1]) for v in VIEWS}, "te", idf=IDF_TE)
    p1 = predict(models1, Fc[FEATS])
    Gc = stage2_feats(Fc, Pc, p1, teRV); del Fc
    p2 = predict(models2, Gc[FEATS2]); del Gc
    keep = apply_rule(Pc, p2, best_cfg, p_ref, y)
    lid = ids_l[lpos]
    for l, r in zip(Pc["li"].values, Pc["ri"].values): cand_te[lid[l]].add(ids_r[r])
    for l, r in zip(Pc["li"].values[keep], Pc["ri"].values[keep]): pred_te[lid[l]].add(ids_r[r])
    n_nonempty = len({l for l in Pc["li"].values[keep]})
    log(f"   {c:8s}: S1={len(Lc):,} pairs={len(Pc):,} ({len(Pc)/len(Lc):.1f}/S1) | predicted non-empty "
        f"{n_nonempty/len(Lc):.3f} | mean matches {keep.sum()/len(Lc):.2f} | {time.time()-t:.0f}s")
    del Pc, p1, p2, keep; gc.collect()

# ============================================================ 11. WRITE SUBMISSION + VALIDATE
def write_ids(path, colname, mapping):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{colname}\n")
        for s in ids_l:
            f.write(f"{s}\t{','.join(sorted(mapping.get(s, set())))}\n")

write_ids(f"{OUT_DIR}/matching_results.tsv", "matched_entity_ids", pred_te)
write_ids(f"{OUT_DIR}/candidate_pairs.tsv", "candidate_entity_ids", cand_te)

valid_r = set(ids_r)
m = read_tsv(f"{OUT_DIR}/matching_results.tsv"); c = read_tsv(f"{OUT_DIR}/candidate_pairs.tsv")
assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
assert m["source1_entity_id"].tolist() == list(ids_l) and m["source1_entity_id"].is_unique
for mm, cc in zip(m["matched_entity_ids"], c["candidate_entity_ids"]):
    a = [x for x in mm.split(",") if x]; b = set(x for x in cc.split(",") if x)
    assert len(a) == len(set(a)) and set(a) <= valid_r and set(a) <= b
log(f"WROTE {OUT_DIR} | S1={len(ids_l):,} | non-empty={sum(1 for x in m.matched_entity_ids if x):,} "
    f"| mean matches={np.mean([len([z for z in x.split(',') if z]) for x in m.matched_entity_ids]):.2f}")
if MODE == "FULL":
    r = subprocess.run([sys.executable, "utils/validate_submission.py",
                        "--matching", f"{OUT_DIR}/matching_results.tsv",
                        "--candidate", f"{OUT_DIR}/candidate_pairs.tsv", "--test-dir", "dataset/test"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-3000:], r.stderr[-2000:])
else:
    log("DEV mode: official validator skipped (DEV output covers only a sample of test S1)")
