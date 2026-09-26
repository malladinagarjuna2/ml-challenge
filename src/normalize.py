"""Text normalization for business names and addresses (US / India / France / unseen countries).

Every function is pure and country-agnostic by default; country-specific tables are applied
only as extra token maps, so an unseen country still gets the generic cleaning.
"""
import re
import unicodedata

from anyascii import anyascii

# ---------------------------------------------------------------- generic text cleaning

NULL_TOKENS = {"null", "none", "nan", "n/a", "na", "-", "--", "unknown", "not available"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_WORD = re.compile(r"\b(?=\w*[a-z])(?=\w*[01345 7@$])[a-z01345 7@$]+\b".replace(" ", ""))

                                                                                                                    
def to_ascii(s: str) -> str:
    """NFKC, transliterate any script to Latin, strip accents, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    return anyascii(s).lower()


def fix_leet(s: str) -> str:
    """'c0mpany' -> 'company', '5ecure' -> 'secure'; leaves pure numbers ('500', 'b-259') alone."""
    def rep(m):
        w = m.group(0)
        letters = sum(c.isalpha() for c in w)
        return w.translate(LEET) if letters >= 2 and letters >= len(w) - 2 else w
    return _LEET_WORD.sub(rep, s)


def base_clean(s: str) -> str:
    if not s or s.strip().lower() in NULL_TOKENS:
        return ""
    malayalam = bool(re.search(r"[ഀ-ൿ]", s))
    s = to_ascii(s)
    if malayalam:  # anyascii renders Malayalam 'ṟṟ' (tt) as 'rr': limirrd -> limittd
        s = s.replace("rr", "tt")
    s = re.sub(r"\bnull\b", " ", s)
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)   # '80th' -> '80', '4th block' -> '4 block' (before leet fix)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", s)     # 'hauts-de-france' -> 'hauts de france'; keeps 'b-59'
    s = s.replace("&", " and ").replace("+", " and ")
    s = fix_leet(s)
    return s


def squash(s: str) -> str:
    s = re.sub(r"[^a-z0-9/\- ]", " ", s)
    s = re.sub(r"(?<![a-z0-9])[-/]|[-/](?![a-z0-9])", " ", s)  # keep '-' '/' only inside tokens (5-513/4)
    return re.sub(r"\s+", " ", s).strip()


def dedup_tokens(toks):
    """'vetsch vetsch hanford' -> 'vetsch hanford' (consecutive repeats only)."""
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# ---------------------------------------------------------------- names

LEGAL = {  # variant -> canonical legal form
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "limited": "ltd", "ltd": "ltd", "llc": "llc", "l l c": "llc",
    "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "private": "pvt", "pvt": "pvt", "pvt ltd": "pvt ltd", "private limited": "pvt ltd", "opc": "opc",
    "sarl": "sarl", "s a r l": "sarl", "sas": "sas", "s a s": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "s a": "sa", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    # Indian-script legal words after transliteration (seen in train: प्राइवेट, பிரைவேட், प्रा. लि., एलएलपी)
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivtt": "pvt", "praivett": "pvt",
    "prayivet": "pvt", "limitet": "ltd", "limittd": "ltd", "limited": "ltd", "limitedd": "ltd",
    "pra li": "pvt ltd", "elelpi": "llp", "kampani": "co", "kampni": "co", "kompani": "co",
}
# multi-word first so 'private limited' wins over 'private'
_LEGAL_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, LEGAL), key=len, reverse=True)) + r")\b")

# descriptor words vendors add/drop freely ("Indchem Power" == "Indchem Services")
GENERIC = {
    "services", "service", "partners", "group", "groupe", "center", "centre", "solutions", "enterprises",
    "enterprise", "industries", "holdings", "international", "global", "and", "the", "of", "de", "du",
    "des", "la", "le", "les", "et", "fils", "freres", "sri", "shri", "shree", "m/s", "ms", "associates",
    "consultants", "trading", "traders", "ventures", "systems", "technologies", "tech", "agency",
}
ALIAS_RE = re.compile(
    r"\b(?:doing business as|d/b/a|dba|formerly known as|formerly|f/k/a|fka|trading as|t/a|a/k/a|aka|"
    r"anciennement|exercant sous)\b")
WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|us|co)\b")
LEAD_JUNK = re.compile(r"^[\W_]+")


def split_camel_web(host: str) -> str:
    # 'wilfordhancock' can't be split without a dictionary; kept as one token (char n-grams handle it)
    return host.replace("-", " ")


def normalize_name(raw: str) -> dict:
    s = base_clean(raw)
    is_web = bool(WEB_RE.search(s))
    s = WEB_RE.sub(lambda m: " " + split_camel_web(m.group(1)) + " ", s)
    s = LEAD_JUNK.sub("", s)

    # alias split: 'x dba y' / 'y formerly x' -> both sides are name variants
    parts = [p for p in ALIAS_RE.split(s) if p.strip()]
    has_alias = len(parts) > 1
    variants = []
    for p in parts:
        p = squash(p)
        p = p.replace("/", " ")
        legal = sorted({LEGAL[m] for m in _LEGAL_RE.findall(p)})
        core = _LEGAL_RE.sub(" ", p)
        toks = dedup_tokens(core.split()) or dedup_tokens(p.split())  # 'Corporation' alone stays a name
        variants.append((" ".join(dedup_tokens(p.split())), " ".join(toks), legal))

    full, core, legal = variants[0] if variants else ("", "", [])
    core_tokens = core.split()
    strict = [t for t in core_tokens if t not in GENERIC] or core_tokens
    return {
        "name": full,                                  # cleaned full name
        "name_core": core,                             # without legal suffixes
        "name_strict": " ".join(strict),               # without legal + generic descriptors
        "name_sorted": " ".join(sorted(strict)),       # word-order invariant key
        "name_alts": "|".join(v[1] for v in variants[1:]),  # alias names (dba / formerly)
        "legal": " ".join(legal),
        "acronym": "".join(t[0] for t in strict if t[0].isalpha()) if len(strict) > 1 else "",
        "is_web": is_web,
        "has_alias": has_alias,
    }


# ---------------------------------------------------------------- addresses

# English-language abbreviations (US + India). Applied AFTER state codes are pulled out, so 'FL', 'CT',
# 'MT', 'NE' at the end of an address stay states instead of becoming floor / court / mount / northeast.
EN_ABBR = {
    "st": "street", "str": "street", "saint": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "anenue": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort",
    # India
    "nr": "near", "opp": "opposite", "opps": "opposite", "sec": "sector", "ph": "phase", "mkt": "market",
    "bldg": "building", "apts": "apartments", "clny": "colony", "ngr": "nagar", "extn": "extension",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk", "po": "post", "ps": "police",
}
FR_ABBR = {  # 'st'/'ste' are saint/sainte in France, never street
    "r": "rue", "av": "avenue", "ave": "avenue", "anenue": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "imp": "impasse", "ch": "chemin", "che": "chemin", "rte": "route", "all": "allee",
    "pl": "place", "fbg": "faubourg", "qu": "quai", "sq": "square", "st": "saint", "ste": "sainte",
    "str": "saint", "street": "saint",
}
ABBR_BY_COUNTRY = {"us": EN_ABBR, "india": EN_ABBR, "france": FR_ABBR}
ANY_ABBR = {**FR_ABBR, **EN_ABBR}  # unseen country: English wins on conflicts
# labels that carry no identity ("H.No 12", "Door No 5", "Plot No 7" -> "12", "5", "7")
ADDR_LABELS = re.compile(r"\b(?:h\s*no|house no|door no|plot no|flat no|shop no|khasra no|kh no|no|number|unit|suite|#)\b\.?")
NOISE_SUFFIX = re.compile(r"\b(cdp|county|city corporation|municipal corporation|mc)\b")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    # transliterated native-script spellings (anyascii output)
    "dilli": "dl", "nii dilli": "dl", "telmgan": "tg", "keralm": "kl", "mharastr": "mh", "maharastr": "mh",
    "krnatk": "ka", "tmilnadu": "tn", "tmil nadu": "tn", "gujrat": "gj", "uttr prdesh": "up",
}
IN_STATE_CODES = set(IN_STATES.values()) | {"ts", "tl", "or", "ut", "uk"}
US_STATE_CODES = set(US_STATES.values())
FR_REGIONS = {  # regions and departments seen in test; kept as region tokens, not required to match
    "nouvelle aquitaine", "hauts de france", "pays de la loire", "ile de france", "occitanie", "bretagne",
    "normandie", "grand est", "auvergne rhone alpes", "provence alpes cote d azur", "gironde", "nord",
    "loire atlantique", "pas de calais", "centre val de loire", "bourgogne franche comte", "corse",
}
STATE_ALIAS = {"tg": "tg", "ts": "tg", "tl": "tg", "or": "od", "ut": "uk"}  # India code aliases

POSTCODE_RE = {"us": re.compile(r"\b(\d{5})(?:-\d{4})?\b"), "india": re.compile(r"\b(\d{3})\s?(\d{3})\b"),
               "france": re.compile(r"\b(\d{5})\b")}
HOUSE_NUM = re.compile(r"\b(?=[a-z0-9\-/]*\d)[a-z]{0,3}[\-]?\d+[a-z0-9\-/]*\b")
LANDMARK_RE = re.compile(r"\b(near|opposite|behind|beside|next to|adjacent to|pres de|en face de)\b[^,]*")


def _state_lookup(text: str, country: str):
    c = country.lower()
    tables = [US_STATES] if c == "us" else [IN_STATES] if c == "india" else [US_STATES, IN_STATES]
    for tbl in tables:
        found = ""
        for full in sorted(tbl, key=len, reverse=True):  # remove every mention ('new delhi ... dilli')
            if re.search(rf"\b{re.escape(full)}\b", text):
                found = found or tbl[full]
                text = re.sub(rf"\b{re.escape(full)}\b", " ", text)
        if found:
            return found, text
    return "", text


def normalize_address(raw: str, country: str = "") -> dict:
    c = (country or "").lower()
    s = base_clean(raw)
    if not s:
        return {"addr": "", "addr_street": "", "nums": "", "postcode": "", "state": "", "region": "",
                "components": "", "has_landmark": False, "landmark": "", "addr_missing": True}
    s = re.sub(r"\(|\)", " ", s)                      # '(41) rue ...' -> '41 rue'

    postcode = ""
    pat = POSTCODE_RE.get(c)
    if pat:
        for m in pat.finditer(s):
            before = s[:m.start()].rstrip()
            if before and not before.endswith(","):  # a leading number is a house number, not a postcode
                postcode = "".join(g for g in m.groups() if g)
                s = s[:m.start()] + " " + s[m.end():]
                break
    s = re.sub(r"\b0+(\d)", r"\1", s)                 # '00709' -> '709' (after zip extraction: '02139')

    # split components on commas (order varies between vendors, so we keep them as a bag)
    comps = [squash(x) for x in s.split(",")]
    comps = [x for x in comps if x]
    text = " , ".join(comps)

    landmark = " ".join(m.group(0).strip() for m in LANDMARK_RE.finditer(text))
    text = ADDR_LABELS.sub(" ", text)
    text = NOISE_SUFFIX.sub(" ", text)

    # states first (before abbreviation expansion): full names anywhere, codes only as the
    # last token of a component ('tilden, tx' / 'erlanger ky') so words like 'in'/'or' are safe
    state, text = _state_lookup(text, c)
    codes = US_STATE_CODES if c == "us" else IN_STATE_CODES if c == "india" else set()
    comps = [x.split() for x in text.split(",")]
    for i, toks in enumerate(comps):
        # ', TX' (whole component) or 'Erlanger KY' (a later, number-free component); never '12 Oak Ct'
        is_code = toks and toks[-1] in codes and (
            len(toks) == 1 or (i > 0 and not any(ch.isdigit() for t in toks for ch in t)))
        if is_code:
            state = state or STATE_ALIAS.get(toks[-1], toks[-1])
            toks.pop()
    region = ""
    if c == "france":
        for r in FR_REGIONS:
            if re.search(rf"\b{r}\b", text):
                region = region or r
                comps = [" ".join(t).replace(r, " ").split() if re.search(rf"\b{r}\b", " ".join(t)) else t
                         for t in comps]

    abbr = ABBR_BY_COUNTRY.get(c, ANY_ABBR)
    text = " , ".join(" ".join(abbr.get(t, t) for t in toks) for toks in comps)

    nums = sorted({n.lstrip("0") or "0" for n in HOUSE_NUM.findall(text) if n != postcode})
    comps = [" ".join(dedup_tokens(x.split())) for x in text.split(",")]
    comps = [x for x in comps if x]
    # per component, so a 'near ...' landmark can't swallow the following components
    street = " ".join(" ".join(t for t in LANDMARK_RE.sub(" ", x).split() if not any(ch.isdigit() for ch in t))
                      for x in comps)
    return {
        "addr": " ".join(" ".join(comps).split()),
        "addr_street": " ".join(street.split()),  # words only, no numbers/landmarks
        "nums": " ".join(nums),
        "postcode": postcode,
        "state": state,
        "region": region,
        "components": "|".join(comps),
        "has_landmark": bool(landmark),
        "landmark": landmark,
        "addr_missing": False,
    }


def normalize_record(name: str, address: str, country: str) -> dict:
    out = normalize_name(name)
    out.update(normalize_address(address, country))
    out["country_n"] = (country or "").strip().lower()
    return out
