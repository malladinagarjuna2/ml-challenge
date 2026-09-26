import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_record

CASES = [
    ("Fannie Trussell Seneca Inc", "TX, 709 Hackberry Street, Tilden", "US"),
    ("Fannie Trussell Seneca Ínc", "709 HACKBERRY ST, TILDEN, TX", "US"),
    ("Fannie Trussell Inc Services", "00709 Hackberry Saint, Tilden, Texas", "US"),
    ("Vetsch, Vetsch, Hanford Hinton Care Services", "1344 MCDONALD HILL ROAD, CHILLICTHE CDP, OH", "US"),
    ("Quoavi Co doing business as Asset Building Committee", "315 80th Street, Chicago, IL", "US"),
    ("Jaxaria Formerly Cabrera Secure Sciences", "165 Barren River Drive, Unit UNIT 2, Erlanger, KY", "US"),
    ("Cabrera 5ecure Sciences LP", "165 BARREN RIVER DR, ERLANGER KY, KY", "US"),
    ("-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC", "US"),
    ("wilfordhancock.com", "Mack Rd, Haltom City, Texas", "US"),
    ("Elite  + C0mpany", "B-59, DERAWAL NAGAR, Delhi", "India"),
    ("LAWRENCE VENTURES PRIVATE", "दिल्ली, JD-36B, PITAMPURA, NEW DELHI", "India"),
    ("Lawrence Ventures Private Limited", "H.no C-956 Jd-36b, New Delhi, दिल्ली", "India"),
    ("Indchem [Power]", "5-513/4, Cbr Estates, Flat No 505 4Th Block, Miya, Pur, Hyderabad, TG", "India"),
    ("ഗാലക്സി ലോജിസ്റ്റിക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "Kerala, ERNAKULAM, FLOOR", "India"),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
    ("Galaxy Logistics Private Limited", "5Th Floor, Seaport Airport Road Near Infopark South Gate, Kakkanad, Kerala", "India"),
    ("<< Team Ecole", "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"),
    ("Europ & Frères Distribution S.A.", "(41) Rue Des Thuyas, Lège-cap-ferret, Gironde", "France"),
    ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
    ("Swing Maison  SARL", "NULL", "France"),
]
KEYS = ["name", "name_strict", "name_alts", "legal", "addr", "addr_street", "nums", "postcode", "state", "landmark"]
for n, a, c in CASES:
    r = normalize_record(n, a, c)
    print(f"\n{n} | {a} [{c}]")
    print("   " + "  ".join(f"{k}={r[k]!r}" for k in KEYS if r[k]))
